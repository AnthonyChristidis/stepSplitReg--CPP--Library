/*
 * ===========================================================
 * File Type: HPP
 * File Name: Model.hpp
 * Package Name: stepSplitReg
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#ifndef Model_hpp
#define Model_hpp

// Libraries included
#include <RcppArmadillo.h>

// Header files included
#include "CV_WEN.hpp"
#include "config.h"

class Model {

    /*
     variables_counter: Variables per model
     variables_in_model: The variables that are currently active in the model
     design_mat: The design matrix that is model specific
     current_H: The current hat matrix based on the variables included in the model
     current_res: The current residuals for each observation basedon the variables included in the model
     current_rss: The current RSS based on the variables included in the model
     */

private:

    // Private variables for each model - Current State
    arma::uvec variables_in_model;
    arma::mat current_design;
    arma::mat current_H;
    arma::vec current_res;
    double current_rss;

    // Private variables to determine when to stop filling a model
    arma::uword max_variables;
    arma::uword stop_criterion;
    double stop_parameter;
    arma::uword variables_counter;
    double R2;
    double aR2;
    double pR2;
    double F_val;
    double p_val;
    arma::uword shrinkage;
    double alpha;
    arma::uword model_type = 1;
    arma::uword include_intercept;
    arma::uword n_lambda_sparsity;
    double tolerance;
    arma::uword max_iter;
    arma::uword n_folds;
    arma::uword n_threads = 1;

    // Private variables for each model - Potential Optimal State
    arma::vec decrease_rss;
    arma::uword optimal_variable;
    double optimal_rss_decrease;

    // Private variable - assessing whether a model is full
    bool model_full;

    // Private variables used in the computation of the model's final state - Regression parameters
    arma::mat final_design;
    double intercept;
    arma::vec betas;
    double cv_error;

public:

    // (+) Model Constructor

    // This constructor initializes the initial number of variables in model to zero
    // Also, also allocating the memory for the number of variables that will be included in each model
    Model(arma::uword max_variables_per_model, const arma::vec& y, 
          const int& stop_criterion, const double& stop_parameter, const arma::uword& number_variables,
          const arma::uword& shrinkage, const double& alpha, const arma::uword& include_intercept,
          const arma::uword& n_lambda_sparsity, const double& tolerance, const arma::uword& max_iter, 
          const arma::uword& n_folds);

    // (+) Functions that update the current state of the model

    // Update the Design matrix
    void UpdateDesign(const arma::mat& x);

    // Update the Hat matrix
    void UpdateH(const arma::vec& y);

    // Update the current residuals
    void UpdateRes(const arma::vec& y);

    // Update the stoping criterion
    void UpdateCriteria(const arma::vec& y);

    // Functions to determine whether the model is full
    void FixedFull();
    void R2Full();
    void aR2Full();
    void pR2Full();
    void FTestFull(const arma::vec& y);

    // Member function to add a variable to the model
    void Variable_Update(int variable_ind, const arma::mat& x, const arma::vec& y);

    // (+) Functions that computes the optimal new variable that may be included in a model 
    
    //     -> Case where this was the optimal model in previous iteration
    void Update_Optimal_Variable_New(arma::uvec candidates, const arma::mat& x, const arma::vec& y,
                                     const bool& initialization);

    //     -> Case where this was NOT the optimal model in previous iteration
    void Update_Optimal_Variable_Check(arma::uvec candidates, const arma::mat& x, const arma::vec& y,
                                       arma::uword previous_optimal_index, arma::uword previous_optimal);

    // (+) Functions that return variables for the current state of the model
    arma::uword Get_Counter();
    void Shed_Variables();
    arma::uvec Get_Variables();
    arma::mat Get_Design();
    arma::mat Get_H();
    arma::vec Get_Res();
    double Get_RSS();

    // (+) Functions that return the final model state
    double Get_Intercept();
    arma::vec Get_Beta();
    double Get_CV_Error();

    // (+) Functions that return variables for the optimal candidate variable of the model
    int Get_Optimal_Variable();
    double Get_Optimal_RSS_Decrease();
    double Get_F_val();
    double Get_p_val();

    // (+) Function that returns a bool value - full model
    bool Get_Full();

    // Function to compute the beta regression vector
    void Set_Final_Design(const arma::mat& x);
    void Compute_Beta(const arma::vec& y);

    // (+) Function to adjust the variable numbers to R convention (start at 1 [C++] rather than 0 [R])
    void Adapt_Variables();

    // (+) Model destructor
    ~Model();
};

#endif // Model_hpp




