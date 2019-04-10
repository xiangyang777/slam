#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>

/****************************
* 本程序演示了 Eigen 几何模块的使用方法
****************************/

int main ( int argc, char** argv ) {

    Eigen::Quaterniond q1(0.55 , 0.3, 0.2, 0.2 );
    Eigen::Quaterniond q2(-0.1 , 0.3, -0.7, 0.2);
    q1=q1.normalized();
    cout<<q1.coeffs() <<endl;

    q2=q2.normalized();

    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();// 虽然称为3d，实质上是4＊4的矩阵
    Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
    T1.rotate(q1);                           // 按照rotation_vector进行旋转
    T2.rotate(q2);
    T1.pretranslate(Eigen::Vector3d(0.7, 1.1, 0.2));                     // 把平移向量设成(1,3,4)
    T2.pretranslate(Eigen::Vector3d(-0.1, 0.4, 0.8));


    Eigen::Vector3d t1(0.7, 1.1, 0.2);
    Eigen::Vector3d t2(-0.1, 0.4, 0.8);

    Eigen::Vector3d v1(0.5, -0.1, 0.2);
    Eigen::Vector4d v11(0.5, -0.1, 0.2,1);
    Eigen::Vector3d v2(1.08228, 0.663509, 0.686957);





    cout<<"T1 * V1= \n"<<T2.matrix()*((T1.matrix()).inverse())*v11<<endl;










}

