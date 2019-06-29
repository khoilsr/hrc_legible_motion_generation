#ifndef KINEMATICS_H
#define KINEMATICS_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>


using namespace Eigen;


class Kinematics
{
private:
	MatrixXd J1(VectorXd& q);
	MatrixXd J2(VectorXd& q);
	MatrixXd J3(VectorXd& q);
	MatrixXd J4(VectorXd& q);
	MatrixXd J5(VectorXd& q);
	MatrixXd J6(VectorXd& q);
	MatrixXd J7(VectorXd& q);
	MatrixXd getJ0(VectorXd& q);

	//values in vector [a, b, c, 1], get multiplied by forward transformation to compute J0 (see Hoffmann paper)
	double a = 0.5;
	double b = 0.03;
	double c = 1;

	int closestJoint = 2;			//number of joint closest to obstacle (from 1 to 7)	
	const double PI = 3.141592653589793238463;				
	Vector3d x0d = Vector3d::Zero();					//velocity of the closest point to the obstacle, needed for null space contraint

public:
	Kinematics();

	void setClosestJoint(int value);
	void setx0d(Vector3d& value);
	//computes pseudo inverse for the jacobian, needed for the angularVelocities, impelemted below
	MatrixXd getPseudoInverse(MatrixXd& J);

	//computes the inverse kinematics with null-space constraints, takes the previous angles and the cartesian velocity and returns the angular velocity
	//VectorXd getCurrentVelocities(MatrixXd& ys, MatrixXd& yds, VectorXd& prevAngles, VectorXd& ts, int t);
	VectorXd getCurrentVelocities(VectorXd& desiredX, VectorXd& desiredXd, VectorXd& q, double timeStep);

	//function computes forward kinematics using ys which includes all joint angles, implemented below
	Vector3d getEndEffectorPosition(VectorXd& q);

	//function computes end effector velocity from joint velocities, uses getJacobian() function, implemented below
	//MatrixXd getEndEffectorVelocity(MatrixXd& ys, MatrixXd& yds, VectorXd& ts);
	VectorXd getEndEffectorVelocity(VectorXd& qd);

	//function to get the Jacobian, implemented below, only used for the getEndEffectorVelocity function
	MatrixXd getJacobian(VectorXd& q);

	//matrix template, for the forward kinematics transformation matrices, input: DH parameters: d, alpha, a, theta
	MatrixXd transformation(double d, double al, double a, double th);
	//computes the forward kinematics for the getEndEffectorVelocity() function
	MatrixXd forwardKinematics(VectorXd& q);

};













#endif