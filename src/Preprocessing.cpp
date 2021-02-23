// 对原始点云利用imu信息进行旋转去畸变，提取点和面特征
#include "utils/common.h"
#include "utils/timer.h"
#include "utils/math_tools.h"

#define  PI  3.1415926535

class Preprocessing {
private:
    // 点云曲率, 40000为一帧点云中点的最大数量
    float cloudCurvature[400000];
    // 曲率点对应的序号
    int cloudSortInd[400000];
    // 点是否筛选过标志：0-未筛选过，1-筛选过
    int cloudNeighborPicked[400000];
    // 点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
    int cloudLabel[400000];
    // 下采样率
    int ds_rate = 2;
    // 采样立方体变长0.6m
    double ds_v = 0.6;
    // cmp函数，比较两个点的曲率
    bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]);}

    ros::NodeHandle nh;

    ros::Subscriber sub_Lidar_cloud;
    ros::Subscriber sub_imu;

    ros::Publisher pub_surf;
    ros::Publisher pub_edge;
    ros::Publisher pub_cutted_cloud;  // 去畸变的点云

    int pre_num = 0;

    pcl::PointCloud<PointType> lidar_cloud_in;
    std_msgs::Header cloud_header;

    vector<sensor_msgs::ImuConstPtr> imu_buf;
    // 待处理的imu数据索引值
    int idx_imu = 0;
    double current_time_imu = -1;
    // 每帧imu数据的角加速度
    Eigen::Vector3d gyr_0;
    Eigen::Quaterniond qIMU = Eigen::Quaterniond::Identity();
    Eigen::Vector3d rIMU = Eigen::Vector3d::Zero();
    bool first_imu = false;
    string imu_topic;

    std::deque<sensor_msgs::PointCloud2> cloud_queue;
    sensor_msgs::PointCloud2 current_cloud_msg;
    double time_scan_next;

    int N_SCANS = 64;

    double qlb0, qlb1, qlb2, qlb3;
    Eigen::Quaterniond q_lb;

    string frame_id = "lili_om_rot";

    // "/points_raw", "/velodyne_pcl_gen/cloud", "/velodyne_points"
    string lidar_topic = "/velodyne_points";

    double runtime = 0;

public:
    Preprocessing():
        nh("~"){

        if (!getParameter("/preprocessing/lidar_topic", lidar_topic)) {
            ROS_WARN("lidar_topic not set, use default value: /velodyne_points");
            lidar_topic = "/velodyne_points";
        }

        if (!getParameter("/preprocessing/line_num", N_SCANS)) {
            ROS_WARN("line_num not set, use default value: 64");
            N_SCANS = 64;
        }

        if (!getParameter("/preprocessing/ds_rate", ds_rate)) {
            ROS_WARN("ds_rate not set, use default value: 1");
            ds_rate = 1;
        }

        if (!getParameter("/common/frame_id", frame_id)) {
            ROS_WARN("frame_id not set, use default value: lili_odom");
            frame_id = "lili_odom";
        }

        if (!getParameter("/backend_fusion/imu_topic", imu_topic)) {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        //extrinsic parameters
        if (!getParameter("/backend_fusion/ql2b_w", qlb0)) {
            ROS_WARN("ql2b_w not set, use default value: 1");
            qlb0 = 1;
        }

        if (!getParameter("/backend_fusion/ql2b_x", qlb1)) {
            ROS_WARN("ql2b_x not set, use default value: 0");
            qlb1 = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_y", qlb2)) {
            ROS_WARN("ql2b_y not set, use default value: 0");
            qlb2 = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_z", qlb3)) {
            ROS_WARN("ql2b_z not set, use default value: 0");
            qlb3 = 0;
        }

        q_lb = Eigen::Quaterniond(qlb0, qlb1, qlb2, qlb3);

        sub_Lidar_cloud = nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic, 100, &Preprocessing::cloudHandler, this);
        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200, &Preprocessing::imuHandler, this);

        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/surf_features", 100);
        pub_edge = nh.advertise<sensor_msgs::PointCloud2>("/edge_features", 100);
        pub_cutted_cloud = nh.advertise<sensor_msgs::PointCloud2>("/lidar_cloud_cutted", 100);
    }

    ~Preprocessing(){}

    // 丢弃一定距离以内的点，遍历把过近的点都丢弃
    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres) {
        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i) {
            if (cloud_in.points[i].x * cloud_in.points[i].x +
                    cloud_in.points[i].y * cloud_in.points[i].y +
                    cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
        if (j != cloud_in.points.size()) {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    template <typename PointT>
    double getDepth(PointT pt) {
        return sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    }

    // 点云中的每个点去畸变
    PointType undistortion(PointType pt, const Eigen::Quaterniond quat) {
        double dt = 0.1;
        int line = int(pt.intensity);
        double dt_i = pt.intensity - line;

        double ratio_i = dt_i / dt;
        if(ratio_i >= 1.0) {
            ratio_i = 1.0;
        }

        Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
        // 根据时间比例取进行球面线性差值，因为弦长相同时，其对应的弧长并不相等
        Eigen::Quaterniond q_si = q0.slerp(ratio_i, qIMU);

        Eigen::Vector3d pt_i(pt.x, pt.y, pt.z);
        // imu转到Lidar坐标系
        q_si = q_lb * q_si * q_lb.inverse();
        // 进行去畸变，内含坐标变化
        Eigen::Vector3d pt_s = q_si * pt_i;

        PointType p_out;
        p_out.x = pt_s.x();
        p_out.y = pt_s.y();
        p_out.z = pt_s.z();
        p_out.intensity = pt.intensity;
        return p_out;
    }

    void solveRotation(double dt, Eigen::Vector3d angular_velocity)
    {
        // 对上一帧和本帧imu角加速度取平均
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity);
        // 由旋转向量转四元数，*=同时也是一个积分的过程
        qIMU *= deltaQ(un_gyr * dt);
        // gyr_0更新
        gyr_0 = angular_velocity;
    }

    void processIMU(double t_cur)
    {
        double rx = 0, ry = 0, rz = 0;
        int i = idx_imu;
        if(i >= imu_buf.size())
            i--;
        // 所有缓存的imu时间戳小于下一帧雷达点云的时间戳的imu都进行处理
        while(imu_buf[i]->header.stamp.toSec() < t_cur) {
            double t = imu_buf[i]->header.stamp.toSec();
            if (current_time_imu < 0)
                current_time_imu = t;
            // 计算两帧imu的时间差（或是与当前雷达帧时间差，因为存在current_time_imu = t_cur;）
            double dt = t - current_time_imu;
            current_time_imu = imu_buf[i]->header.stamp.toSec();

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;
            solveRotation(dt, Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }
        // 已缓存的imu数据还有但是时间戳已大于下一帧雷达点云的时间戳
        if(i < imu_buf.size()) {
            // dt1为下一帧雷达点云的时间戳与current_time_imu的时间差
            double dt1 = t_cur - current_time_imu;                   // current_time_imu | t_cur | imu_buf[i]->header.stamp.toSec()
            // dt2为与刚好大于下一帧雷达点云的时间戳的imu时间差
            double dt2 = imu_buf[i]->header.stamp.toSec() - t_cur;
            // 计算两个时间差所占的比例
            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            // 计算更新在dt1的旋转四元数
            solveRotation(dt1, Eigen::Vector3d(rx, ry, rz));
        }
        current_time_imu = t_cur;
        idx_imu = i;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn)
    {
        // 缓存指向imu数据的指针
        imu_buf.push_back(ImuIn);
        // 如果size太大就把最前边的赋为空指针
        if(imu_buf.size() > 600)
            imu_buf[imu_buf.size() - 601] = nullptr;
        // 未收到imu数据时，current_time_imu==-1
        if (current_time_imu < 0)
            current_time_imu = ImuIn->header.stamp.toSec();

        if (!first_imu)
        {
            first_imu = true;
            double rx = 0, ry = 0, rz = 0;
            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            // 第一帧imu的角加速度
            gyr_0 = angular_velocity;
        }
    }


    void cloudHandler( const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        cloud_queue.push_back(*laserCloudMsg);
        if (cloud_queue.size() <= 2)
            return;
        else {
            current_cloud_msg = cloud_queue.front();
            cloud_queue.pop_front();
            // cloud_header赋值为点云队列头部点云header
            cloud_header = current_cloud_msg.header;
            cloud_header.frame_id = frame_id;
            // time_scan_next赋值为下一帧待处理点云时间戳
            time_scan_next = cloud_queue.front().header.stamp.toSec();
        }

        int tmpIdx = 0;
        if(idx_imu > 0)
            tmpIdx = idx_imu - 1;
        // 此帧lidar无相应时间戳内的imu数据
        if (imu_buf.empty() || imu_buf[tmpIdx]->header.stamp.toSec() > time_scan_next) {
            ROS_WARN("Waiting for IMU data ...");
            return;
        }

        // LidarPreprocessing开始计时
        Timer t_pre("LidarPreprocessing");
        // 记录每个scan有曲率的点的开始和结束索引
        std::vector<int> scanStartInd(N_SCANS, 0);
        std::vector<int> scanEndInd(N_SCANS, 0);

        pcl::PointCloud<PointType> lidar_cloud_in;
        pcl::fromROSMsg(current_cloud_msg, lidar_cloud_in);
        std::vector<int> indices;
        // 移除空点和以及在Lidar坐标系原点MINIMUM_RANGE距离以内的点
        pcl::removeNaNFromPointCloud(lidar_cloud_in, lidar_cloud_in, indices);
        removeClosedPointCloud(lidar_cloud_in, lidar_cloud_in, 3.0);
        // 点云点的数量
        int cloudSize = lidar_cloud_in.points.size();

        // atan2计算的是原点到(x,y)点与想轴的夹角，逆时针为正，顺时针为负
        // lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
        float startOri = -atan2(lidar_cloud_in.points[0].y, lidar_cloud_in.points[0].x);
        // lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi
        float endOri = -atan2(lidar_cloud_in.points[cloudSize - 1].y,
                              lidar_cloud_in.points[cloudSize - 1].x) + 2 * M_PI;
        // Pandar40:[-2.70111,8.90765]->[-158,510]->[-158,150]，即便是角度不对应好像也不影响
        // cout << startOri << " " << endOri << endl;
        // 结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描，正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
        if (endOri - startOri > 3 * M_PI)
            endOri -= 2 * M_PI;
        else if (endOri - startOri < M_PI)
            endOri += 2 * M_PI;

        // 如果已经收到了第一帧imu数据，对imu数据进行处理
        if(first_imu)
            processIMU(time_scan_next);
        if(isnan(qIMU.w()) || isnan(qIMU.x()) || isnan(qIMU.y()) || isnan(qIMU.z())) {
            qIMU = Eigen::Quaterniond::Identity(); // 计算的当前帧雷达旋转四元数出错则重置
        }
        // lidar扫描线是否旋转过半
        bool halfPassed = false;
        int count = cloudSize;
        PointType point;
        PointType point_undis;
        // 每一行扫描视为一个点云
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
        for (int i = 0; i < cloudSize; i++) {
            point.x = lidar_cloud_in.points[i].x;
            point.y = lidar_cloud_in.points[i].y;
            point.z = lidar_cloud_in.points[i].z;
            point.intensity = 0.1 * lidar_cloud_in.points[i].intensity;

            // 计算点的俯仰角
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (N_SCANS == 16) {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            }
            else if (N_SCANS == 32) {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            }
            // cxp添加20210210
            else if (N_SCANS == 40) {
            if (angle >= 2)
                scanID = int((angle + 32.0) + 0.5);
            else if (angle < 2 && angle >= -6)
                scanID = int((3.0 * angle + 28.0) + 0.5);
            else
                scanID = int((angle + 16.0) + 0.5);
            if (angle > 7 || angle < -16 || scanID > (N_SCANS - 1) || scanID < 0) {
                count--;
                continue;
                }
            }
            else if (N_SCANS == 64) {
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                    count--;
                    continue;
                }
            }
            else {
                printf("wrong scan number\n");
                ROS_BREAK();
            }
            // 获取每一个点对应的水平角度
            float ori = -atan2(point.y, point.x);
            // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
            if (!halfPassed) {
                if (ori < startOri - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > startOri + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - startOri > M_PI)
                    halfPassed = true;
            }
            else {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > endOri + M_PI / 2)
                    ori -= 2 * M_PI;
            }
            // -0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
            float relTime = (ori - startOri) / (endOri - startOri);
            // 点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
            point.intensity = scanID + 0.1 * relTime;
            // 由qIMU对每个点进行去畸变
            point_undis = undistortion(point, qIMU);
            // 把当前点放在对应的scanID对应的线上
            laserCloudScans[scanID].push_back(point_undis);
        }

        cloudSize = count;
        // printf("points size %d \n", cloudSize);

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCANS; i++) {
            // 再将每条线单独的点云合成一个大的点云，且设置好每一条线的开始和结束索引
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
        }


        for (int i = 5; i < cloudSize - 5; i++) {
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
            cloudSortInd[i] = i;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
        }

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;

        for (int i = 0; i < N_SCANS; i++) {
            if( scanEndInd[i] - scanStartInd[i] < 6 || i % ds_rate != 0)
                continue;
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            for (int j = 0; j < 6; j++) {
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                auto bound_comp = bind(&Preprocessing::comp, this, _1, _2);
                std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, bound_comp);

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] > 2.0) {

                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp.push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 10) {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else
                            break;

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSortInd[k];

                    if(laserCloud->points[ind].x*laserCloud->points[ind].x+laserCloud->points[ind].y*laserCloud->points[ind].y+laserCloud->points[ind].z*laserCloud->points[ind].z < 0.25)
                        continue;

                    if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] < 0.1) {

                        cloudLabel[ind] = -1;
                        surfPointsFlat.push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                            break;

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if(laserCloud->points[k].x*laserCloud->points[k].x+laserCloud->points[k].y*laserCloud->points[k].y+laserCloud->points[k].z*laserCloud->points[k].z < 0.25)
                        continue;
                    if (cloudLabel[k] <= 0)
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
            // 由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(ds_v, ds_v, ds_v);
            downSizeFilter.filter(surfPointsLessFlatScanDS);

            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }

        sensor_msgs::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = current_cloud_msg.header.stamp;
        laserCloudOutMsg.header.frame_id = frame_id;
        pub_cutted_cloud.publish(laserCloudOutMsg);

        sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = current_cloud_msg.header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = frame_id;
        pub_edge.publish(cornerPointsLessSharpMsg);

        sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = current_cloud_msg.header.stamp;
        surfPointsLessFlat2.header.frame_id = frame_id;
        pub_surf.publish(surfPointsLessFlat2);

        qIMU = Eigen::Quaterniond::Identity();
        rIMU = Eigen::Vector3d::Zero();
        //t_pre.tic_toc();
        runtime += t_pre.toc();
        //cout<<"pre_num: "<<++pre_num<<endl;
        //cout<<"Preprocessing average run time: "<<runtime / pre_num<<endl;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lili_om_rot");
    Preprocessing Pre;
    ROS_INFO("\033[1;32m---->\033[0m Preprocessing Started.");

    ros::spin();
    return 0;
}
