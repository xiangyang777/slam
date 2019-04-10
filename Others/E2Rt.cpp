//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    Eigen::JacobiSVD<Eigen::MatrixX3d> svd(E,ComputeFullU|ComputeFullV);
    Eigen::Matrix3d U=svd.matrixU();
    Eigen::Matrix3d V=svd.matrixV();
    Eigen::Vector3d A=svd.singularValues();


    Eigen::Matrix3d M =Eigen::Matrix3d.setZero();

    M(0,0)=(A(0,0)+A(0,1))/2;
    M(1,1)=M(0,0);

    Eigen::AngleAxisd Rz(M_PI_2,Vector3d(0,0,1));
    Eigen::Matrix3d RZ1=Rz.toRotationMatrix();
    Eigen::AngleAxisd Rz1(-M_PI_2,Vector3d(0,0,1));
    Eigen::Matrix3d RZ2=Rz1.toRotationMatrix();



    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    Matrix3d t_wedge1;
    Matrix3d t_wedge2;

    Matrix3d R1;
    Matrix3d R2;

    t_wedge1=U*RZ1*M*V.transpose();
    R1=U*RZ1*M*U.transpose();
    R2=U*RZ2*M*U.transpose();
    // END YOUR CODE HERE

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3d::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}