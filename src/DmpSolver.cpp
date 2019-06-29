#include "DmpSolver.hpp"


DmpSolver::DmpSolver() 
{

}

DmpSolver::DmpSolver(double tau, VectorXd& start, VectorXd& goal, double dt, Trajectory& traj, VectorXd& weightInput, double covarSize, int nBasisFunc, bool useTraj, double alphaGoal, double alphaSpringDamper, double eliteness, double covarDecayFactor, double integrate_dmp_beyond_tau_factor) 
	:_tau(tau), _start(start), _goal(goal), _dt(dt), _covarSize(covarSize), _alphaGoal(alphaGoal), _nBasisFunc(nBasisFunc), _alphaSpringDamper(alphaSpringDamper), 
	_integrate_dmp_beyond_tau_factor(integrate_dmp_beyond_tau_factor), _nDims(goal.size())
{
	VectorXd one  = VectorXd::Ones(1);
 	VectorXd zero = VectorXd::Zero(1);	
	_goalSys   = new ExponentialSystem(tau, start, goal, _alphaGoal);
	_phaseSys  = new TimeSystem(tau);
	_gatingSys = new ExponentialSystem(tau, one, zero, 5);				//starts at one, ends at 0, decay const = 5, ensures convergence, 1D

	int timeSteps = ceil(tau*integrate_dmp_beyond_tau_factor/dt)+1;	//number of time steps 
	_ts = VectorXd::LinSpaced(timeSteps, 0.0, _tau);			//vector with all time steps
	//_trajectory = Trajectory::generateMinJerkTrajectory(_ts, _start, _goal);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//center, width and weight of the basic functions
	VectorXd centers = VectorXd::LinSpaced(_nBasisFunc,0,tau*0.8);
    VectorXd widths  = VectorXd::Constant(_nBasisFunc,tau*0.8/4.3);
    VectorXd  weights  = VectorXd::Zero(_nBasisFunc);
    double intersection_height = 0.5;   //intersection height

    MatrixXd weightsMat = MatrixXd::Zero(_nDims, _nBasisFunc);

    if(weightInput(0,0) != 0) {		//if we are reading the weight values from a file
    	for(int i = 0; i<_nDims; i++) {
    	//for(int i = 0; i < _nBasisFunc; i++) {
    		//weightsMat.block(0, i, _nDims, 1) = weightInput.segment(i*(_nDims-1), _nDims);	
    		weightsMat.block(i, 0, 1, _nBasisFunc) = weightInput.segment(i*(_nBasisFunc), _nBasisFunc).transpose();
    		cout << "reading weights " << i << endl;
    		cout << weightsMat << endl;
    	}
    }

    //metaParametersRBFN and modelParametersRBFN for distributing the basic functions
    ModelParametersRBFN* model_parameters[_nDims];
    FunctionApproximatorRBFN *faRbfn[_nDims];
    for(int i = 0; i<_nDims; i++) {
		model_parameters[i] = new ModelParametersRBFN(centers,widths, weightsMat.block(i, 0, 1, _nBasisFunc).transpose());
		faRbfn[i] = new FunctionApproximatorRBFN(model_parameters[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//	_metaParam = new MetaParametersRBFN(1, _nBasisFunc);
	//_faRbfn = new FunctionApproximatorRBFN(_metaParam, model_parameters);
 //    vector<FunctionApproximator*> _funcAppr;
 //    _faRbfn = new FunctionApproximatorRBFN(model_parameters);

	// for (int i = 0; i < _nDims; i++) {
 //        _funcAppr.push_back(_faRbfn);
	// }

	if(useTraj) {	//initialize with meta parameters
		_metaParam = new MetaParametersRBFN(1,_nBasisFunc,intersection_height);
		//_metaParam = new MetaParametersRBFN(1, _nBasisFunc);
		FunctionApproximatorRBFN* faRbfn_meta = new FunctionApproximatorRBFN(_metaParam);
		for (int i = 0; i < _nDims; i++) {
        	_funcAppr.push_back(faRbfn_meta);
		}
		_dmp = new Dmp(_nDims, _funcAppr, _alphaSpringDamper, _goalSys, _phaseSys, _gatingSys);
		//_dmp = new Dmp(_tau, _start, _goal, _funcAppr, _alphaSpringDamper, _goalSys, _phaseSys, _gatingSys);
		cout << "Dmp object generated. " << endl;
		_dmp->train(traj);			//train dmp with the given trajectory (loaded from file)
		cout << "Training complete. " << endl;
	}
	else {			//initialize with model parameters
		for (int i = 0; i < _nDims; i++) {
       	 _funcAppr.push_back(faRbfn[i]);
		}
		_dmp = new Dmp(_tau, _start, _goal, _funcAppr, _alphaSpringDamper, _goalSys, _phaseSys, _gatingSys);
	}


	set<string> parameters_to_optimize;
    parameters_to_optimize.insert("weights");		//we optimize the weights of the RBFN
    _taskSolver = new TaskSolverDmp(_dmp, parameters_to_optimize, _dt, _integrate_dmp_beyond_tau_factor, false);

    _dmp->getParameterVectorSelected(_meanInit);	//-> centers around the weights
    MatrixXd covarInit = _covarSize * MatrixXd::Identity(_meanInit.size(), _meanInit.size());
    ///////////////////////////////////////////change weight of the rollout/////////////////////////////
    //cout << "Covariance matrix size: " << _meanInit.size() << endl;
    // MatrixXd temp_weight = MatrixXd::Identity(_meanInit.size(), _meanInit.size());
    // temp_weight(0,0) = 1;
    // temp_weight(1,1) = 0.5;
    // temp_weight(2,2) = 0.5;
    // MatrixXd covarInit = temp_weight;
    ///////////////////////////////////////////change weight of the rollout/////////////////////////////
    _distribution = new DistributionGaussian(_meanInit, covarInit);		//distribution around the parameters to sample

    string weightingMethod("PI-BB");	//BB: reward-weighted averaging
    _updater = new UpdaterCovarDecay(_eliteness, _covarDecayFactor, weightingMethod);
}

void DmpSolver::generateSamples(int nSamplesPerUpdate, MatrixXd& samples){
	_distribution->generateSamples(nSamplesPerUpdate, samples);
	//cout << "samples " << endl << samples << endl;
}

void DmpSolver::performRollouts(MatrixXd& samples, MatrixXd& costVariables) {
		//cout << "columns costVars " << costVariables.cols() << ", rows costVars " << costVariables.rows() << endl;
	_taskSolver->performRollouts(samples, costVariables);			//costVariables now includes y_out, yd_out, ydd_out, ts, forcing terms for each sample
}

void DmpSolver::updateDistribution(MatrixXd& samples, VectorXd& costPerSample) {
	_updater->updateDistribution(*_distribution, samples, costPerSample, *_distribution);
}

//output matrix dimensions: nDims x (centers+widths+weights)
VectorXd DmpSolver::getCentersWidthsAndWeights() {
	// MatrixXd output(_nDims, 3*_nBasisFunc);
	// for(int i = 0; i < _nDims; i++) {
	// 	const ModelParametersRBFN* model_param = static_cast<const ModelParametersRBFN*>(_funcAppr[i]->getModelParameters());
	// 	//VectorXd parameters;
	// 	//model_param->getParameterVectorAll(parameters);
	// 	//_dmp->getParameterVectorSelected(parameters, false);
	// 	output.block(i, 0, 1, output.cols()) = parameters.transpose();	//save model parameters from all dimensions in one matrix
	// }
	VectorXd parameters = _distribution->mean(); 	//nBasisFunc*nDims
	//return output;
	return parameters.transpose();
}

DmpSolver::~DmpSolver() {
	delete _goalSys;
	delete _phaseSys;
	delete _gatingSys;
	delete _metaParam;
	//delete _funcAppr;
	//delete _faRbfn;
	delete _dmp;
	//delete _taskSolver;
	delete _distribution;
	delete _updater;
}


