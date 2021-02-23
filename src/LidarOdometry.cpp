// 主要完成根据scan2localmap匹配发布里程计数据,且只用了surf特征
#include "utils/common.h"
#include "utils/math_tools.h"
#include "utils/timer.h"
#include "factors/LidarKeyframeFactor.h"

class LidarOdometry {
private:
    int odom_pub_cnt = 0;
    ros::NodeHandle nh;

    ros::Subscriber sub_edge;
    ros::Subscriber sub_surf;
    ros::Subscriber sub_full_cloud;

    ros::Publisher pub_edge;
    ros::Publisher pub_surf;
    ros::Publisher pub_full_cloud;
    ros::Publisher pub_odom;
    ros::Publisher pub_each_odom;
    ros::Publisher pub_path;

    std_msgs::Header cloud_header;                  // 预处理节点订阅的最新去畸变点云header(预处理赋值为原始点云header)->即原始点云header
    nav_msgs::Odometry odom;
    nav_msgs::Path path;

    pcl::PointCloud<PointType>::Ptr edge_features;  // 从预处理节点订阅的角点特征点云
    pcl::PointCloud<PointType>::Ptr surf_features;  // 从预处理节点订阅的面特征点云
    pcl::PointCloud<PointType>::Ptr full_cloud;     // 从预处理节点订阅的去畸变点云

    pcl::PointCloud<PointType>::Ptr surf_last_ds;   // 最新收到的surf_features下采样后的点云

    pcl::KdTreeFLANN<PointType >::Ptr kd_tree_surf_last;

    /// Lidar odom的原点是：第一帧laser时刻的laser坐标系，求得的位姿是laser帧在odom下的位姿
    /// 不同于后端，后端的原点map是：第一帧imu坐标系，求得的位姿是关键帧时刻的imu在map下的位姿
    // pose representation: [quaternion: w, x, y, z | transition: x, y, z]
    double abs_pose[7];   //absolute pose from current frame to the first frame
    double rel_pose[7];   //relative pose between two frames

    bool new_edge = false;                          // 从预处理节点订阅的角点特征点云标志
    bool new_surf = false;                          // 从预处理节点订阅的面特征点云标志
    bool new_full_cloud = false;                    // 从预处理节点订阅的去畸变点云标志

    double time_new_surf = 0;
    double time_new_full_points = 0;
    double time_new_edge = 0;

    bool system_initialized;

    int surf_res_cnt;

    int max_num_iter;
    int scan_match_cnt;

    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_cloud_frame; //pose of each frame
    pcl::PointCloud<PointXYZI>::Ptr pose_cloud_frame; //position of each frame

    vector<pcl::PointCloud<PointType>::Ptr> surf_frames;  // 接收到的surf特征进行下采样后的点云队列，Lidar局部坐标系
    deque<pcl::PointCloud<PointType>::Ptr> recent_surf_frames; // 由surf_frames转换为的odom全局坐标系

    pcl::PointCloud<PointType>::Ptr surf_from_map;      // 由surf特征建立的局部子图
    pcl::PointCloud<PointType>::Ptr surf_from_map_ds;   // 下采样后的局部子图

    pcl::PointCloud<PointType>::Ptr surf_current_pts;
    pcl::PointCloud<PointType>::Ptr surf_normal;

    pcl::VoxelGrid<PointType> down_size_filter_surf;
    pcl::VoxelGrid<PointType> down_size_filter_surf_map;

    int latest_frame_idx;

    bool kf = true;             // 是否为关键帧的标志
    int kf_num = 0;             // 当前或上一帧pose_cloud_frame的size

    Eigen::Vector3d trans_last_kf = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat_last_kF = Eigen::Quaterniond::Identity();

    string frame_id = "lili_om_rot";
    bool if_to_deskew;
    double runtime = 0;

public:
    LidarOdometry(): nh("~") {
        initializeParameters();
        allocateMemory();

        sub_edge = nh.subscribe<sensor_msgs::PointCloud2>("/edge_features", 100, &LidarOdometry::laserCloudLessSharpHandler, this);
        sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/surf_features", 100, &LidarOdometry::laserCloudLessFlatHandler, this);
        sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/lidar_cloud_cutted", 100, &LidarOdometry::FullPointCloudHandler, this);

        pub_edge = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
        pub_odom = nh.advertise<nav_msgs::Odometry>("/odom", 100);
        pub_each_odom = nh.advertise<nav_msgs::Odometry>("/each_odom", 100);
        pub_path = nh.advertise<nav_msgs::Path>("/path", 100);
        pub_full_cloud = nh.advertise<sensor_msgs::PointCloud2>("/full_point_cloud", 100);
    }

    ~LidarOdometry(){}

    void allocateMemory() {
        edge_features.reset(new pcl::PointCloud<PointType>());
        surf_features.reset(new pcl::PointCloud<PointType>());
        full_cloud.reset(new pcl::PointCloud<PointType>());

        kd_tree_surf_last.reset(new pcl::KdTreeFLANN<PointType>());

        pose_info_cloud_frame.reset(new pcl::PointCloud<PointPoseInfo>()); //pose of each frame
        pose_cloud_frame.reset(new pcl::PointCloud<PointXYZI>()); //position of each frame

        surf_from_map.reset(new pcl::PointCloud<PointType>());
        surf_from_map_ds.reset(new pcl::PointCloud<PointType>());
        surf_last_ds.reset(new pcl::PointCloud<PointType>());

        surf_current_pts.reset(new pcl::PointCloud<PointType>());
        surf_normal.reset(new pcl::PointCloud<PointType>());
    }

    void initializeParameters() {
        // Load parameters from yaml
        if (!getParameter("/common/frame_id", frame_id)) {
            ROS_WARN("frame_id not set, use default value: lili_om");
            frame_id = "lili_om";
        }

        if (!getParameter("/lidar_odometry/if_to_deskew", if_to_deskew)) {
            ROS_WARN("if_to_deskew not set, use default value: true");
            if_to_deskew = true;
        }

        if (!getParameter("/lidar_odometry/max_num_iter", max_num_iter)) {
            ROS_WARN("maximal iteration number not set, use default value: 50");
            max_num_iter = 15;
        }

        if (!getParameter("/lidar_odometry/scan_match_cnt", scan_match_cnt)) {
            ROS_WARN("number of scan matching not set, use default value: 1");
            scan_match_cnt = 1;
        }

        latest_frame_idx = 0;

        odom.header.frame_id = frame_id;

        system_initialized = false;

        abs_pose[0] = 1;
        rel_pose[0] = 1;

        for (int i = 1; i < 7; ++i) {
            abs_pose[i] = 0;
            rel_pose[i] = 0;
        }
        surf_res_cnt = 0;

        down_size_filter_surf.setLeafSize(0.4, 0.4, 0.4);
        down_size_filter_surf_map.setLeafSize(0.4, 0.4, 0.4);
    }

    void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &pointCloudIn) {
        time_new_edge = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *edge_features);
        new_edge = true;
    }

    void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &pointCloudIn) {
        time_new_surf = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *surf_features);
        new_surf = true;
    }

    void FullPointCloudHandler(const sensor_msgs::PointCloud2ConstPtr &pointCloudIn) {
        time_new_full_points = pointCloudIn->header.stamp.toSec();
        cloud_header = pointCloudIn->header;
        pcl::fromROSMsg(*pointCloudIn, *full_cloud);
        new_full_cloud = true;
    }

    void undistortion(const pcl::PointCloud<PointType>::Ptr &pcloud, const Eigen::Vector3d trans, const Eigen::Quaterniond quat) {
        double dt = 0.1;
        for (auto &pt : pcloud->points) {
            int line = int(pt.intensity);
            double dt_i = pt.intensity - line;
            double ratio_i = dt_i / dt;

            if(ratio_i > 1)
                ratio_i = 1;

            Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
            Eigen::Quaterniond q_si = q0.slerp(ratio_i, quat);

            Eigen::Vector3d t_si = ratio_i * trans;
            Eigen::Vector3d pt_i(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_s = q_si * pt_i + t_si;

            pt.x = pt_s.x();
            pt.y = pt_s.y();
            pt.z = pt_s.z();
        }
    }

    void checkInitialization() {
        sensor_msgs::PointCloud2 msgs;
        pcl::toROSMsg(*edge_features, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_edge.publish(msgs);

        pcl::toROSMsg(*surf_features, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_surf.publish(msgs);

        pcl::toROSMsg(*full_cloud, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_full_cloud.publish(msgs);

        system_initialized = true;
    }

    // 点：雷达局部坐标系点全局坐标系即转到相对于起点的坐标系
    void transformPoint(PointType const *const pi, PointType *const po) {
        Eigen::Quaterniond quaternion(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);
        Eigen::Vector3d transition(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);

        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    // 点云：雷达局部坐标系转全局坐标系即转到相对于起点的坐标系
    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, PointPoseInfo * PointInfoIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        Eigen::Quaterniond quaternion(PointInfoIn->qw,
                                      PointInfoIn->qx,
                                      PointInfoIn->qy,
                                      PointInfoIn->qz);
        Eigen::Vector3d transition(PointInfoIn->x,
                                   PointInfoIn->y,
                                   PointInfoIn->z);

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);
        for (int i = 0; i < numPts; ++i)
        {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            cloudOut->points[i] = pt;
        }
        return cloudOut;
    }

    // 由之前的最多20帧数据建立的局部子图，未包括最新雷达帧因为要使用scan2map求解last帧位姿
    void buildLocalMap() {
        surf_from_map->clear();
        // Initialization, localmap直接赋值为surf_features
        if (pose_cloud_frame->points.size() <= 1) { // 初始化时加入一个空点,即第2帧laser
            //ROS_INFO("Initialization for odometry local map");
            *surf_from_map += *surf_features;
            return;
        }

        // If already more then 20 frames, pop the frames at the beginning
        if (recent_surf_frames.size() < 20) {
            int i = pose_cloud_frame->points.size() - 1;
            recent_surf_frames.push_back(transformCloud(surf_frames[i], &pose_info_cloud_frame->points[i]));
        } else { // 如果已经超过了20帧且surf_frames最新一帧未处理
            if (latest_frame_idx != pose_cloud_frame->points.size() - 1) {
                recent_surf_frames.pop_front();
                latest_frame_idx = pose_cloud_frame->points.size() - 1;
                recent_surf_frames.push_back(transformCloud(surf_frames[latest_frame_idx], &pose_info_cloud_frame->points[latest_frame_idx]));
            }
        }

        for (int i = 0; i < recent_surf_frames.size(); ++i)
            *surf_from_map += *recent_surf_frames[i];
    }

    void clearCloud() {
        surf_from_map->clear();
        surf_from_map_ds->clear();
        edge_features->clear();
        surf_features->clear();
        full_cloud->clear();
        if(surf_frames.size() > 7)
            surf_frames[surf_frames.size() - 8]->clear();
    }

    void downSampleCloud() {
        // surf_from_map_ds相当于下采样了两次，一次是对surf_features下采样，一次是对构建的子图整体进行下采样
        down_size_filter_surf_map.setInputCloud(surf_from_map);
        down_size_filter_surf_map.filter(*surf_from_map_ds);

        surf_last_ds->clear();

        down_size_filter_surf.setInputCloud(surf_features);
        down_size_filter_surf.filter(*surf_last_ds);
    }

    void savePoses() {
        PointXYZI tmpPose;
        tmpPose.x = abs_pose[4];
        tmpPose.y = abs_pose[5];
        tmpPose.z = abs_pose[6];
        tmpPose.intensity = pose_cloud_frame->points.size();
        pose_cloud_frame->push_back(tmpPose);

        PointPoseInfo tmpPoseInfo;
        tmpPoseInfo.x = abs_pose[4];
        tmpPoseInfo.y = abs_pose[5];
        tmpPoseInfo.z = abs_pose[6];
        tmpPoseInfo.qw = abs_pose[0];
        tmpPoseInfo.qx = abs_pose[1];
        tmpPoseInfo.qy = abs_pose[2];
        tmpPoseInfo.qz = abs_pose[3];
        tmpPoseInfo.idx = pose_cloud_frame->points.size();
        tmpPoseInfo.time = time_new_surf;
        pose_info_cloud_frame->push_back(tmpPoseInfo);

        pcl::PointCloud<PointType>::Ptr surfEachFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*surf_last_ds, *surfEachFrame);

        surf_frames.push_back(surfEachFrame);
    }

    // 特征关联，构造残差项
    void findCorrespondingSurfFeatures() {
        surf_res_cnt = 0;
        // 遍历当前帧的平面点
        for (int i = 0; i < surf_last_ds->points.size(); ++i) {
            PointType point_sel;
            // 将点变换到lidar odom全局坐标系
            transformPoint(&surf_last_ds->points[i], &point_sel);
            vector<int> point_search_idx;
            vector<float> point_search_dists;
            // 从局部地图的kd-tree中找到5个最近点
            kd_tree_surf_last->nearestKSearch(point_sel, 5, point_search_idx, point_search_dists);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();

            if (point_search_dists[4] < 1.0) { // 如果最远的第五个点距离也是小于1m
                PointType center;

                for (int j = 0; j < 5; ++j) {
                    matA0(j, 0) = surf_from_map_ds->points[point_search_idx[j]].x;
                    matA0(j, 1) = surf_from_map_ds->points[point_search_idx[j]].y;
                    matA0(j, 2) = surf_from_map_ds->points[point_search_idx[j]].z;
                }

                // 直接求解出平面法向量 Get the norm of the plane using linear solver based on QR composition 求解matA0*norm = matB0
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double normInverse = 1 / norm.norm(); // norm()为求解模长
                norm.normalize(); // get the unit norm 向量归一化为单位向量

                // Compute the centroid of the plane 计算平面的质心
                center.x = matA0.col(0).sum() / 5.0;
                center.y = matA0.col(1).sum() / 5.0;
                center.z = matA0.col(2).sum() / 5.0;

                // Make sure that the plane is fit 确保平面合适
                bool planeValid = true;
                for (int j = 0; j < 5; ++j) {
                    if (fabs(norm.x() * surf_from_map_ds->points[point_search_idx[j]].x +
                             norm.y() * surf_from_map_ds->points[point_search_idx[j]].y +
                             norm.z() * surf_from_map_ds->points[point_search_idx[j]].z + normInverse) > 0.06) {
                        planeValid = false;
                        break;
                    }
                }

                // if one eigenvalue is significantly larger than the other two 如果一个特征值明显大于另外两个
                if (planeValid) {
                    float pd = norm.x() * point_sel.x + norm.y() * point_sel.y + norm.z() * point_sel.z + normInverse;
                    float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(point_sel.x * point_sel.x + point_sel.y * point_sel.y + point_sel.z * point_sel.z));
                    // 判断平面有效性，根据平面的分布计算点的权重
                    if(weight > 0.4) {
                        PointType normal;
                        normal.x = weight * norm.x();
                        normal.y = weight * norm.y();
                        normal.z = weight * norm.z();
                        normal.intensity = weight * normInverse;
                        surf_current_pts->push_back(surf_last_ds->points[i]);
                        surf_normal->push_back(normal);
                        ++surf_res_cnt;
                    }
                }
            }
        }
    }

    // 当前帧绝对位姿初始化
    void poseInitialization() {
        // q0上一帧的位姿
        Eigen::Quaterniond q0(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);
        Eigen::Vector3d t0(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);
        // dq上一帧相对于上上帧的位姿变换
        Eigen::Quaterniond dq(rel_pose[0],
                rel_pose[1],
                rel_pose[2],
                rel_pose[3]);
        Eigen::Vector3d dt(rel_pose[4],
                rel_pose[5],
                rel_pose[6]);
        // 假定last帧的位姿变换与上一帧与上上帧之间的位姿变换相同，初始化位姿
        t0 = q0 * dt + t0;
        q0 = q0 * dq;

        // abs_pose坐标系为x前，y左，z上(即Lidar坐标系)
        // cout << abs_pose[0] << " " << abs_pose[1] << " " << abs_pose[2] << " " << abs_pose[3] << " " << abs_pose[4]<< " "
             // << abs_pose[5] << " " << abs_pose[6] << endl;

        abs_pose[0] = q0.w();
        abs_pose[1] = q0.x();
        abs_pose[2] = q0.y();
        abs_pose[3] = q0.z();

        abs_pose[4] = t0.x();
        abs_pose[5] = t0.y();
        abs_pose[6] = t0.z();
    }

    void computeRelative() {
        Eigen::Quaterniond quaternion1;
        Eigen::Vector3d transition1;
        if(pose_info_cloud_frame->points.empty()) {
            quaternion1 = Eigen::Quaterniond::Identity();
            transition1 = Eigen::Vector3d::Zero();
        } else {
            int max_idx = pose_info_cloud_frame->points.size();
            quaternion1 = Eigen::Quaterniond(pose_info_cloud_frame->points[max_idx-2].qw,
                    pose_info_cloud_frame->points[max_idx-2].qx,
                    pose_info_cloud_frame->points[max_idx-2].qy,
                    pose_info_cloud_frame->points[max_idx-2].qz);
            transition1 = Eigen::Vector3d (pose_info_cloud_frame->points[max_idx-2].x,
                    pose_info_cloud_frame->points[max_idx-2].y,
                    pose_info_cloud_frame->points[max_idx-2].z);
        }

        Eigen::Quaterniond quaternion2(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);
        Eigen::Vector3d transition2(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);


        Eigen::Quaterniond quaternion_r = quaternion1.inverse() * quaternion2;
        Eigen::Vector3d transition_r = quaternion1.inverse() *(transition2 - transition1);

        rel_pose[0] = quaternion_r.w();
        rel_pose[1] = quaternion_r.x();
        rel_pose[2] = quaternion_r.y();
        rel_pose[3] = quaternion_r.z();
        rel_pose[4] = transition_r.x();
        rel_pose[5] = transition_r.y();
        rel_pose[6] = transition_r.z();
    }

    // 仅使用平面点来配准
    void updateTransformationWithCeres() {
        // Make sure there is enough feature points in the sweep
        if (surf_from_map_ds->points.size() < 10) {
            ROS_WARN("Not enough feature points from the map");
            return;
        }
        // 设置surf_from_map_ds为索引的范围
        kd_tree_surf_last->setInputCloud(surf_from_map_ds);

        double transformInc[7] = {abs_pose[0],
                                  abs_pose[1],
                                  abs_pose[2],
                                  abs_pose[3],
                                  abs_pose[4],
                                  abs_pose[5],
                                  abs_pose[6]}; // 当前帧init pose

        int match_cnt; // 迭代次数
        if(pose_info_cloud_frame->points.size() < 2)
            match_cnt = 8;
        else
            match_cnt = scan_match_cnt; // scan_match_cnt从yaml中读取默认为1

        for (int iter_cnt = 0; iter_cnt < match_cnt; iter_cnt++) {
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(0.1);
            ceres::LocalParameterization *quatParameterization = new ceres:: QuaternionParameterization();
            ceres::Problem problem;
            // 添加待求解参数
            problem.AddParameterBlock(transformInc, 4, quatParameterization);
            problem.AddParameterBlock(transformInc + 4, 3);
            // 特征关联
            findCorrespondingSurfFeatures();
            // 遍历特征关联,构造残差项
            for (int i = 0; i < surf_res_cnt; ++i) {
                Eigen::Vector3d currentPt(surf_current_pts->points[i].x,
                                          surf_current_pts->points[i].y,
                                          surf_current_pts->points[i].z);
                Eigen::Vector3d norm(surf_normal->points[i].x,
                                     surf_normal->points[i].y,
                                     surf_normal->points[i].z);
                double normInverse = surf_normal->points[i].intensity;
                // 构造因子
                ceres::CostFunction *costFunction = LidarPlaneNormIncreFactor::Create(currentPt, norm, normInverse);
                // 添加残差项 (cost func， 核函数 ， 参数块1, 参数块2)
                problem.AddResidualBlock(costFunction, lossFunction, transformInc, transformInc + 4);
            }

            ceres::Solver::Options solverOptions;
            solverOptions.linear_solver_type = ceres::DENSE_QR;
            solverOptions.max_num_iterations = max_num_iter;
            solverOptions.max_solver_time_in_seconds = 0.015;
            solverOptions.minimizer_progress_to_stdout = false;
            solverOptions.check_gradients = false;
            solverOptions.gradient_check_relative_precision = 1e-2;

            ceres::Solver::Summary summary;
            ceres::Solve( solverOptions, &problem, &summary );

            // 实部>=0, shortest quaternion
            if(transformInc[0] < 0) {
                Eigen::Quaterniond tmpQ(transformInc[0],
                        transformInc[1],
                        transformInc[2],
                        transformInc[3]);
                tmpQ = unifyQuaternion(tmpQ);
                transformInc[0] = tmpQ.w();
                transformInc[1] = tmpQ.x();
                transformInc[2] = tmpQ.y();
                transformInc[3] = tmpQ.z();
            }
            //每次迭代完，清空
            surf_current_pts->clear();
            surf_normal->clear();
            // 更新位姿
            abs_pose[0] = transformInc[0];
            abs_pose[1] = transformInc[1];
            abs_pose[2] = transformInc[2];
            abs_pose[3] = transformInc[3];
            abs_pose[4] = transformInc[4];
            abs_pose[5] = transformInc[5];
            abs_pose[6] = transformInc[6];
        }

        // 关键帧选取
        //    double ratio_u = double(surfResCount) / double(surfLastDS->points.size());
        Eigen::Vector3d transCur = Eigen::Vector3d(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);
        Eigen::Quaterniond quatCur = Eigen::Quaterniond(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);

        double dis = (transCur - trans_last_kf).norm();                   // 距离
        double ang = 2 * acos((quat_last_kF.inverse() * quatCur).w());  // 四元数到轴角的近似公式->角度
        // 满足以下条件判断为关键帧，后续通过kf标志决定是否发布里程计数据
        if(((dis > 0.2 || ang > 0.1) && (pose_cloud_frame->points.size() - kf_num > 1) || (pose_cloud_frame->points.size() - kf_num > 2)) || pose_cloud_frame->points.size() <= 1){
            kf = true;
            trans_last_kf = Eigen::Vector3d(abs_pose[4],
                    abs_pose[5],
                    abs_pose[6]);
            quat_last_kF = Eigen::Quaterniond(abs_pose[0],
                    abs_pose[1],
                    abs_pose[2],
                    abs_pose[3]);
        } else
            kf = false;
    }

    // 发布全局odom
    void publishOdometry() {
        odom.header.stamp = cloud_header.stamp;
        odom.pose.pose.orientation.w = abs_pose[0];
        odom.pose.pose.orientation.x = abs_pose[1];
        odom.pose.pose.orientation.y = abs_pose[2];
        odom.pose.pose.orientation.z = abs_pose[3];
        odom.pose.pose.position.x = abs_pose[4];
        odom.pose.pose.position.y = abs_pose[5];
        odom.pose.pose.position.z = abs_pose[6];
        pub_odom.publish(odom);
//        坐标系为:x前,y左,z上(Lidar坐标系)
//        cout << abs_pose[0] << " " << abs_pose[1] << " " << abs_pose[2] << " " << abs_pose[3] << " " << abs_pose[4]<< " "
//             << abs_pose[5] << " " << abs_pose[6] << endl;

        geometry_msgs::PoseStamped poseStamped;
        poseStamped.header = odom.header;
        poseStamped.pose = odom.pose.pose;
        poseStamped.header.stamp = odom.header.stamp;
        path.header.stamp = odom.header.stamp;
        path.poses.push_back(poseStamped);
        path.header.frame_id = frame_id;
        pub_path.publish(path);
    }

    // 发布相对的odom
    void publishEachOdometry() {
        nav_msgs::Odometry eachOdom;
        eachOdom.header.frame_id = frame_id;

        eachOdom.header.stamp = cloud_header.stamp;
        eachOdom.pose.pose.orientation.w = rel_pose[0];
        eachOdom.pose.pose.orientation.x = rel_pose[1];
        eachOdom.pose.pose.orientation.y = rel_pose[2];
        eachOdom.pose.pose.orientation.z = rel_pose[3];
        eachOdom.pose.pose.position.x = rel_pose[4];
        eachOdom.pose.pose.position.y = rel_pose[5];
        eachOdom.pose.pose.position.z = rel_pose[6];
        pub_each_odom.publish(eachOdom);
    }

    void publishCloudLast() {
        Eigen::Vector3d trans(rel_pose[4], rel_pose[5], rel_pose[6]);
        Eigen::Quaterniond quat(1, 0, 0, 0);

        if(if_to_deskew) {
            undistortion(surf_features, trans, quat);
            undistortion(edge_features, trans, quat);
            undistortion(full_cloud, trans, quat);
        }

        sensor_msgs::PointCloud2 msgs;

        pcl::toROSMsg(*edge_features, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_edge.publish(msgs);

        pcl::toROSMsg(*surf_features, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_surf.publish(msgs);

        pcl::toROSMsg(*full_cloud, msgs);
        msgs.header.stamp = cloud_header.stamp;
        msgs.header.frame_id = frame_id;
        pub_full_cloud.publish(msgs);
    }

    void run() {
        // 证明收到一帧从预处理节点发来的数据(标志+时间 校验)
        if (new_surf && new_full_cloud && new_edge
                && abs(time_new_full_points - time_new_surf) < 0.1
                && abs(time_new_full_points - time_new_edge) < 0.1) {
            new_surf = false;
            new_edge = false;
            new_full_cloud = false;
        } else
            return;
        // checkInitialization完成丢弃第一帧，设置system_initialized为true
        if (!system_initialized) {
            savePoses(); // pose_cloud_frame、pose_info_cloud_frame、surf_frames加入空点
            checkInitialization();
            return;
        }
        // 当前帧绝对位姿初始化，
        poseInitialization();
        Timer t_odm("LidarOdometry");
        // 提取的surf特征，建立局部子图，最多20帧
        buildLocalMap();
        // 对上述建立的局部子图和surf_features进行下采样
        downSampleCloud();
        updateTransformationWithCeres();
        savePoses();
        // 计算相对位姿
        computeRelative();
        //// 只有判断为关键帧才进行后端优化
        if(kf) {
            kf_num = pose_cloud_frame->points.size();
            publishOdometry();
            publishCloudLast();
        }
        publishEachOdometry();
        clearCloud();
        //cout<<"odom_pub_cnt: "<<++odom_pub_cnt<<endl;
        //t_odm.tic_toc();
        runtime += t_odm.toc();
        //cout<<"Odometry average run time: "<<runtime / odom_pub_cnt<<endl;
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    ros::init(argc, argv, "lili_om_rot");

    LidarOdometry LO;

    ROS_INFO("\033[1;32m---->\033[0m Lidar Odometry Started.");

    ros::Rate rate(200);

    while (ros::ok()) {
        ros::spinOnce();
        LO.run();
        rate.sleep();   // 根据之前ros::Rate的定义来控制发布话题的频率
    }

    ros::spin();
    return 0;
}