/*
 * ===========================================================
 * File Type: CPP
 * File Name: StepSplitReg_Main.cpp
 * Package Name: StepSplitReg
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

 // Libraries included
#include <RcppArmadillo.h>
#include <iostream>

// Header files included
#include "Model.hpp"
#include "Model_Functions.hpp"
#include "Set_Diff.hpp"
#include "config.h"

// [[Rcpp::export]]
Rcpp::List Stepwise_Split(arma::mat x,
                          arma::vec y,
                          arma::uword n_models,
                          arma::uword max_variables_per_model,
                          const arma::uword& model_criterion,
                          const arma::uword& stop_criterion,
                          const double& stop_parameter,
                          const arma::uword& shrinkage, const double& alpha, const arma::uword& include_intercept,
                          const arma::uword& n_lambda, const double& tolerance, const arma::uword& max_iter,
                          const arma::uword& n_folds) {

    // Standardizing the covariates
    arma::rowvec mu_x = mean(x);
    arma::rowvec sd_x = stddev(x, 1);
    // New scaled design matrix
    arma::mat x_std = x;
    x_std.each_row() -= mu_x;
    x_std.each_row() /= sd_x;
    // New centered response vector
    arma::vec y_c = y;
    double mu_y = mean(y);
    y_c = y - mu_y;

    // Storing the number of variables
    arma::uword n_var = x_std.n_cols;

    // Create the memory for the models (through dynamic allocation)
    Model* models[n_models];
    // Initialize the models through the constructors
    for (arma::uword m = 0; m < n_models; m++)
    {
        models[m] = new Model(max_variables_per_model, y_c, 
                              stop_criterion, stop_parameter, x_std.n_cols,
                              shrinkage, alpha, include_intercept,
                              n_lambda, tolerance, max_iter,
                              n_folds);
    }

    // Let's initialize the candidates
    arma::uvec candidates = arma::linspace<arma::uvec>(0, x_std.n_cols - 1, x_std.n_cols);

    // Variables used to store the optimal model
    arma::vec models_rss_decrease(n_models);
    arma::vec models_p_val(n_models);
    arma::uword optimal_model;

    // Variable to store index of optimal variable (for the optimal model) from the candidates
    arma::uvec index_optimal(1);

    // Variable to store the number of full models
    arma::uword full_models;

    // Initializing the optimal variable for the models
    for (arma::uword m = 0; m < n_models; m++) {
        models[m]->Update_Optimal_Variable_New(candidates, x_std, y_c, true);
    }

    // Variable to store number of variables used
    arma::uword n_var_iter = 0;

    // Let's see which of the variables is the best candidate
    do {

        if (model_criterion == 1) {

            // Find the model with the optimal decrease in RSS
            for (arma::uword m = 0; m < n_models; m++) {
                if (!(models[m]->Get_Full()))
                    models_rss_decrease(m) = models[m]->Get_Optimal_RSS_Decrease();
                else
                    models_rss_decrease(m) = -1;
            }
            // Optimal model 
            optimal_model = models_rss_decrease.index_max();
        }
        else if (model_criterion == 2) {
            // Find the model with the optimal decrease in RSS
            for (arma::uword m = 0; m < n_models; m++) {
                if (!(models[m]->Get_Full()))
                    models_p_val(m) = models[m]->Get_p_val();
                else
                    models_p_val(m) = -1;
            }
            // Optimal model 
            optimal_model = models_p_val.index_min();
        }

        // Add the best variable for the optimal model
        models[optimal_model]->Variable_Update(models[optimal_model]->Get_Optimal_Variable(), x_std, y_c);
        // Remove the optimal variable from the candidates
        index_optimal = arma::find(candidates == models[optimal_model]->Get_Optimal_Variable(), 1);
        candidates.shed_row(index_optimal(0));

        // Updating the optimal variable for the non-optimal model(s)
        // Parallelization over the models? Not recommended to multi-thread in and out of R.
        for (arma::uword m = 0; m < n_models; m++)
            if (m != optimal_model && !(models[m]->Get_Full()))
                models[m]->Update_Optimal_Variable_Check(candidates, x_std, y_c, index_optimal(0), models[optimal_model]->Get_Optimal_Variable());
        // Updating the optimal variable for the optimal model
        if (!models[optimal_model]->Get_Full())
            models[optimal_model]->Update_Optimal_Variable_New(candidates, x_std, y_c, false);

        // Computing the number of full models
        full_models = 0;
        for (arma::uword m = 0; m < n_models; m++) {
            full_models += models[m]->Get_Full();
        }

        // Iterating the numbers of variables used
        n_var_iter++;

    } while (!(full_models == n_models) && (n_var_iter + 1) < n_var);
    
    // Let's compute the regression parameter vector for each mdoel
    for (arma::uword m = 0; m < n_models; m++) {
        models[m]->Set_Final_Design(x);
        models[m]->Compute_Beta(y);
    }

    // Computing the variables with all the information for the models
    Rcpp::List final_intercepts = Generate_Inctercept_List(models, n_models);
    Rcpp::List final_betas = Generate_Beta_List(models, n_models);
    Rcpp::List final_cv_error = Generate_CV_List(models, n_models);

    // Let's adjust the variable indicators for R notation
    for (arma::uword m = 0; m < n_models; m++) {
        models[m]->Adapt_Variables();
    }
    // We return the variables in each model
    // arma::umat final_variables = Generate_Variables(models, n_models, max_variables_per_model);
    Rcpp::List final_variables = Generate_Variables_List(models, n_models);

    // Destroy the models
    for (arma::uword m = 0; m < n_models; m++) {
        delete(models[m]);
    }

    // Returning final state of ensemble
    Rcpp::List final_ensemble;
    final_ensemble["variables"] = final_variables;
    final_ensemble["intercepts"] = final_intercepts;
    final_ensemble["betas"] = final_betas;
    final_ensemble["cv_error"] = final_cv_error;
    return(final_ensemble);
}

// --------------------------
// --------------------------
// Cross-Validation Function
// --------------------------
// --------------------------

// [[Rcpp::export]]
Rcpp::List CV_Stepwise_Split(arma::mat x,
                             arma::vec y,
                             const arma::vec& n_models,
                             arma::uword& max_variables_per_model,
                             const arma::uword& model_criterion,
                             const arma::uword& stop_criterion,
                             const double& stop_parameter,
                             const arma::uword& shrinkage, const double& alpha, const arma::uword& include_intercept,
                             const arma::uword& n_lambda, const double& tolerance, const arma::uword& max_iter,
                             const arma::uword& n_folds,
                             const arma::uword& n_threads){
    
    // Storing the number of variables and observations
    const arma::uword n = x.n_rows;
    
    // Creating indices for the folds of the data
    const arma::uvec indin = arma::linspace<arma::uvec>(0, n - 1, n);
    const arma::uvec inint = arma::linspace<arma::uvec>(0, n, n_folds + 1);
    
    // Variables to store CV MSPE
    arma::vec cv_mspe = arma::zeros(n_models.n_elem);
    
    // Looping over the number of models
    for(arma::uword model_ind=0; model_ind < n_models.n_elem; model_ind++){
        
        // Current model size
        arma::uword n_models_cv = n_models[model_ind];
        
        // Split Stepwise over the folds (with parallelization)
        # pragma omp parallel for num_threads(n_threads)
        for (arma::uword fold = 0; fold < n_folds; fold++) {
            
            // Get test and training samples
            arma::uvec test = arma::linspace<arma::uvec>(inint[fold], inint[fold + 1] - 1, inint[fold + 1] - inint[fold]);
            arma::uvec train = Set_Diff(indin, test);
            arma::uword max_variables_fold = ((max_variables_per_model < train.n_elem) ? max_variables_per_model : train.n_elem);
            
            // Running the algorithm for the fold
            Rcpp::List cv_output = Stepwise_Split(x.rows(train), y.rows(train),
                                                  n_models_cv,
                                                  max_variables_fold,
                                                  model_criterion,
                                                  stop_criterion,
                                                  stop_parameter,
                                                  shrinkage, alpha, include_intercept,
                                                  n_lambda, tolerance, max_iter,
                                                  n_folds);
            
            // Lists for data from the mdoels
            Rcpp::List variables_list = cv_output["variables"];
            Rcpp::List intercepts_list = cv_output["intercepts"];
            Rcpp::List betas_list = cv_output["betas"];
            
            // Vector to store the predictions 
            arma::vec predictions = arma::zeros(test.n_elem);
            
            for (arma::uword model_ind = 0; model_ind < n_models.n_elem; model_ind++){
                
                // Extracting the variables in the model
                Rcpp::NumericVector rcpp_variables = variables_list[model_ind];
                arma::vec arma_variables = arma::vec(rcpp_variables);
                arma::uvec arma_uvariables = arma::conv_to<arma::uvec>::from(arma_variables);
                arma_uvariables = arma_uvariables - 1;
                
                // Extracting the intercept
                double arma_intercept = intercepts_list[model_ind];
                
                // Extracting the betas
                Rcpp::NumericVector rcpp_betas = betas_list[model_ind];
                arma::vec arma_betas = arma::vec(rcpp_betas);
                
                // Predictions
                predictions += arma_intercept + x.submat(test, arma_uvariables) * arma_betas;
            }
            
            // Average of predictions
            predictions /= n_models_cv;
            
            // Storing the test fold indices
            std::cout << "Fold " << fold << ": " << arma::mean(arma::square(y.rows(test) - predictions)) << std::endl;
            cv_mspe[model_ind] += arma::mean(arma::square(y.rows(test) - predictions));
        }
    }
    
    // Adjusting CV MSPE
    cv_mspe /= n_folds;
    
    // Optimal number of models
    arma::uword n_models_optimal = n_models[cv_mspe.index_min()];
    
    // Stepwise split with optimal number of models
    // Running the algorithm for the fold
    Rcpp::List optimal_output = Stepwise_Split(x, y,
                                               n_models_optimal,
                                               max_variables_per_model,
                                               model_criterion,
                                               stop_criterion,
                                               stop_parameter,
                                               shrinkage, alpha, include_intercept,
                                               n_lambda, tolerance, max_iter,
                                               n_folds);
    
    // Returning the full output
    return(optimal_output);
}

