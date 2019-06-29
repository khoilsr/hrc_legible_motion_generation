#ifndef DMPSOLVER_H
#define DMPSOLVER_H

#include "dmp_bbo_node/TaskSolver.hpp"
#include "dmp_bbo_node/TaskSolverDmp.hpp"
#include "dmp_bbo_node/Task.hpp"
#include "dmp_bbo_node/TaskViapoint.hpp"
#include "dmp_bbo_node/Trajectory.hpp"
#include "dmp_bbo_node/Dmp.hpp"
#include "dmp_bbo_node/ModelParametersLWR.hpp"
#include "dmp_bbo_node/FunctionApproximatorLWR.hpp"
#include "dmp_bbo_node/ModelParametersRBFN.hpp"
#include "dmp_bbo_node/MetaParametersRBFN.hpp"
#include "dmp_bbo_node/FunctionApproximatorRBFN.hpp"
#include "dmp_bbo_node/DistributionGaussian.hpp"
#include "dmp_bbo_node/Updater.hpp"
#include "dmp_bbo_node/UpdaterCovarDecay.hpp"
#include "dmp_bbo_node/SpringDamperSystem.hpp"
#include "dmp_bbo_node/ExponentialSystem.hpp"
#include "dmp_bbo_node/TimeSystem.hpp"
#include "dmp_bbo_node/SigmoidSystem.hpp"
#include "dmp_bbo_node/EigenFileIO.hpp"
#include "dmp_bbo_node/UpdateSummary.hpp"
#include "dmp_bbo_node/runEvolutionaryOptimization.hpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>


#include <map>
#include <limits>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <vector>
#include <string>
#include <set>

using namespace Eigen;
using namespace DmpBbo;
using namespace std;

class DmpSolver 
{
private:
	DmpSolver();
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & _goalSys;
		ar & _phaseSys;
		ar & _gatingSys;
		ar & _trajectory;
		ar & _metaParam;
		//ar & _faRbfn;
		ar & _dmp;
		ar & _taskSolver;
		ar & _distribution;
		ar & _updater;
	    ar & _tau;
	    ar & _start;
	    ar & _goal;
	    ar & _dt;
	    ar & _meanInit;
	    ar & _covarSize;
	    ar & _alphaGoal;
	    ar & _nBasisFunc;
	    ar & _alphaSpringDamper;
	    ar & _integrate_dmp_beyond_tau_factor;
	    ar & _ts;
	    ar & _nDims;
	    ar & _eliteness;
	    ar & _covarDecayFactor;
	    ar & _meanCost;
	}
public: 
	DmpSolver(double tau, VectorXd& start, VectorXd& goal, double dt, Trajectory& traj, VectorXd& weightInput, double covarSize=300, int nBasisFunc=20, bool useTraj=false, double alphaGoal=15, double alphaSpringDamper=20, double eliteness=10, double covarDecayFactor=0.83, double integrate_dmp_beyond_tau_factor=1.35);	//constructor
	~DmpSolver(); 	//destructor
	void generateSamples(int nSamplesPerUpdate, MatrixXd& samples);
	void performRollouts(MatrixXd& samples, MatrixXd& costVariables);
	void updateDistribution(MatrixXd& samples, VectorXd& costPerSample);
	VectorXd getCentersWidthsAndWeights();

	DynamicalSystem *_goalSys;
	DynamicalSystem *_phaseSys;
	DynamicalSystem *_gatingSys;
	Trajectory _trajectory;
	MetaParametersRBFN *_metaParam;
	vector<FunctionApproximator*> _funcAppr;
	//FunctionApproximatorRBFN *_faRbfn;
	Dmp *_dmp;
	TaskSolverDmp *_taskSolver;
	DistributionGaussian *_distribution;
	Updater *_updater;

	double _tau;
	VectorXd _start;
	VectorXd _goal;
	double _dt;
	VectorXd _meanInit;
	// !!CHANGING THE VALUES HERE WILL NOT CHANGE ANYTHING, change in the constructor or in the main function
	double _covarSize = 300;	//the bigger the variance, the bigger the sample space
	double _alphaGoal = 15;		//before: 20 decay constant
	int _nBasisFunc = 30;      //before: 3 number of Gaussian function per trajectory
	double _alphaSpringDamper = 20;
	double _integrate_dmp_beyond_tau_factor = 1.35;  //before 1.2
	VectorXd _ts;	//vector of time steps
	int _nDims = 3;
	double _eliteness = 10;				//values from old code
    double _covarDecayFactor = 0.83;		//sample space gets smaller with time
    double _meanCost;
};

#endif
