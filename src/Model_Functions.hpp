/*
 * ===========================================================
 * File Type: HPP
 * File Name: Model_Functions.hpp
 * Package Name: stepSplitReg
 *
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */


// Libraries included
#include <RcppArmadillo.h>
#include <vector>

// Header files included
#include "config.h"
#include "Model.hpp"

// Return a list of vectors with the variables in each model
Rcpp::List Generate_Variables_List(std::vector<Model*> final_models,
                                   const arma::uword& n_models) {

    Rcpp::List final_variables(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_variables[m] = final_models[m]->Get_Variables();

    return(final_variables);
}

// Return a list of design matrices for each model
Rcpp::List Generate_Design_List(std::vector<Model*> final_models,
                                const arma::uword& n_models) {

    Rcpp::List final_design(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_design[m] = final_models[m]->Get_Design();

    return(final_design);
}

// Return the design matrix for each model
Rcpp::List Generate_H_List(std::vector<Model*> final_models,
                           const arma::uword& n_models) {

    Rcpp::List final_H(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_H[m] = final_models[m]->Get_H();

    return(final_H);
}

// Return the Res for each model
Rcpp::List Generate_Res_List(std::vector<Model*> final_models,
                             const arma::uword& n_models) {

    Rcpp::List final_res(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_res[m] = final_models[m]->Get_Res();

    return(final_res);
}

// Return the RSS for each model
Rcpp::List Generate_RSS_List(std::vector<Model*> final_models,
                             const arma::uword& n_models) {

    Rcpp::List final_rss(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_rss[m] = final_models[m]->Get_RSS();

    return(final_rss);
}

// Return the intercepts for each model
Rcpp::List Generate_Inctercept_List(std::vector<Model*> final_models,
                                    const arma::uword& n_models) {

    Rcpp::List final_intercepts(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_intercepts[m] = final_models[m]->Get_Intercept();

    return(final_intercepts);
}

// Return the coefficients for each model
Rcpp::List Generate_Beta_List(std::vector<Model*> final_models,
                              const arma::uword& n_models) {

    Rcpp::List final_betas(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_betas[m] = final_models[m]->Get_Beta();

    return(final_betas);
}

// Return the CV errors for each model
Rcpp::List Generate_CV_List(std::vector<Model*> final_models,
                            const arma::uword& n_models) {

    Rcpp::List final_cv_error(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_cv_error[m] = final_models[m]->Get_CV_Error();

    return(final_cv_error);
}
