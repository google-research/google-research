# Copyright 2021 The On Combining Bags to Better Learn from Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code for testing LogReg, MM, LMM(v(G,s)), AMM(MM), AMM(LMM(v(G,s)) using API in github.com/giorgiop/almostnolabel/blob/master/demo.R
# Bag distribution Scenario III
library(doMC)
registerDoMC(cores = 2)
library(ggplot2)
library(data.table)
library(digest)
library(pROC)
library(R.utils)

almostnolabel_master <- "./almostnolabel-master/"

source(file = paste(almostnolabel_master, "auc.R", sep = ""))
source(file = paste(almostnolabel_master, "mean.map.R", sep = ""))
source(file = paste(almostnolabel_master, "laplacian.mean.map.R", sep = ""))
source(file = paste(almostnolabel_master, "alternating.mean.map.R", sep = ""))
source(file = paste(almostnolabel_master, "logistic.regression.R", sep = ""))

root_for_experiments <- "./Data/"


names_list <- list("Australian", "Ionosphere", "Heart")



# Hyperparameters List
lambda_list <- list(0, 1, 10, 100)
gamma_list <- list(0.01, 0.1, 1)
sigma_list <- list(0.25, 0.5, 1)



for (name in names_list) {
  name_dir <- paste(root_for_experiments, name, "/", sep = "")



  clusterBagsMethod <- 3

  # create file for writing result lines

  clusterBagsMethodoutfile <- paste(name_dir, name, "RexpOutputClusterBags_", clusterBagsMethod, sep = "")

  for (foldnumber in 1:5) {
    folddir <- paste(name_dir, "Fold_", foldnumber, "/", sep = "")

    for (splitnumber in 1:5) {
      splitdir <- paste(folddir, "Split_", splitnumber, "/", sep = "")

      orig_test_data_file <- paste(splitdir, name, "_", foldnumber, "_", splitnumber, "-test.csv", sep = "")

      orig_train_data_file <- paste(splitdir, name, "_", foldnumber, "_", splitnumber, "-train.csv", sep = "")

      extended_train_data_file <- paste(splitdir, "ClusterBags_", clusterBagsMethod, "/full_train.csv", sep = "")

      orig_test.data <- read.csv(orig_test_data_file)
      orig_test.data <- orig_test.data[sample(nrow(orig_test.data)), ]


      orig_train.data <- read.csv(orig_train_data_file)
      orig_train.data <- orig_train.data[sample(nrow(orig_train.data)), ]

      extended_train.data <- read.csv(extended_train_data_file)
      extended_train.data <- extended_train.data[sample(nrow(extended_train.data)), ]

      orig_train_set <- orig_train.data

      trainset <- extended_train.data

      testset <- orig_test.data

      N <- length(unique(trainset$bag)) # count the bags into the trainset

      results_line <- paste(name, clusterBagsMethod, splitnumber, foldnumber, sep = ",")

      print(results_line)


      print(paste("Started: Name = ", name, "  Method = ", clusterBagsMethod, "  Split = ", splitnumber, "  Fold = ", foldnumber, sep = ""))

      # Logistic regression - the labels are the binary ones, not proportions - Oracle
      #

      set.seed(247392)

      for (lambda in lambda_list) {
        print(paste("Doing Logistic for lambda = ", lambda))

        tryCatch(
          {
            w.lr <- withTimeout(
              {
                logistic.regression(orig_train_set, lambda)
              },
              timeout = 3600
            )
            test.X <- as.matrix(testset[, -c(1, 2)])
            test.pred <- 1 / (1 + exp(-2 * test.X %*% w.lr))
            test.auc <- auc((testset$label + 1) / 2, test.pred)
            print(test.auc)
            results_line <<- paste(results_line, test.auc, sep = " , ")
          },
          TimeoutException = function(ex) {
            print("Timeout Occurred")
            results_line <<- paste(results_line, "timeout", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          },
          error = function(e) {
            print("Error Occurred")
            results_line <<- paste(results_line, "error", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          }
        )
        print(results_line)
      }

      # Cast label in proportions
      for (bag in unique(trainset$bag)) {
        id <- which(trainset$bag == bag)
        trainset$label[id] <- rep(mean((trainset$label[id] + 1) / 2), length(id))
      }

      # Mean Map
      for (lambda in lambda_list) {
        print(paste("Doing MM for lambda = ", lambda))


        tryCatch(
          {
            withTimeout(
              {
                w.mm <- mean.map(trainset, lambda)
              },
              timeout = 3600
            )
            test.X <- as.matrix(testset[, -c(1, 2)])
            test.pred <- 1 / (1 + exp(-2 * test.X %*% w.mm))
            test.auc <- auc((testset$label + 1) / 2, test.pred)
            print(test.auc)
            results_line <<- paste(results_line, test.auc, sep = " , ")
          },
          TimeoutException = function(ex) {
            print("Timeout Occurred")
            results_line <<- paste(results_line, "timeout", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          },
          error = function(e) {
            print("Error Occurred")
            results_line <<- paste(results_line, "error", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          }
        )
        print(results_line)
      }

      # Laplacian Mean Map with similarity v^{G,s}

      for (lambda in lambda_list) {
        for (gamma in gamma_list) {
          for (sigma in sigma_list) {
            print(paste("Doing LMM for lambda,gamma,sigma = ", lambda, gamma, sigma))

            tryCatch(
              {
                withTimeout(
                  {
                    L <- laplacian(similarity = "G,s", trainset, N, sigma = sigma)
                  },
                  timeout = 3600
                )
                w.lmm <- laplacian.mean.map(trainset, lambda, gamma, L = L)
                test.X <- as.matrix(testset[, -c(1, 2)])
                test.pred <- 1 / (1 + exp(-2 * test.X %*% w.lmm))
                test.auc <- auc((testset$label + 1) / 2, test.pred)
                print(test.auc)
                results_line <<- paste(results_line, test.auc, sep = " , ")
              },
              TimeoutException = function(ex) {
                print("Timeout Occurred")
                results_line <<- paste(results_line, "timeout", sep = " , ")
                write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
                print(results_line)
                # print("Done this")
              },
              error = function(e) {
                print("Error Occurred")
                results_line <<- paste(results_line, "error", sep = " , ")
                write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
                print(results_line)
                # print("Done this")
              }
            )
            print(results_line)
          }
        }
      }

      # Alternating Mean Map started with MM

      for (lambda in lambda_list) {
        print(paste("Doing AMM-MM for lambda = ", lambda))


        tryCatch(
          {
            withTimeout(
              {
                w.amm <- alternating.mean.map(trainset, lambda = lambda, init = "MM")
                w.amm <- w.amm$theta # the algorithm returns a structure that contains also the number of step until termination
              },
              timeout = 3600
            )
            test.X <- as.matrix(testset[, -c(1, 2)])
            test.pred <- 1 / (1 + exp(-2 * test.X %*% w.amm))
            test.auc <- auc((testset$label + 1) / 2, test.pred)
            print(test.auc)
            results_line <<- paste(results_line, test.auc, sep = " , ")
          },
          TimeoutException = function(ex) {
            print("Timeout Occurred")
            results_line <<- paste(results_line, "timeout", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          },
          error = function(e) {
            print("Error Occurred")
            results_line <<- paste(results_line, "error", sep = " , ")
            write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
            print(results_line)
            # print("Done this")
          }
        )
        print(results_line)
      }

      # Alternating Mean Map started with LMM with similarity v^{G,s}

      for (lambda in lambda_list) {
        for (gamma in gamma_list) {
          print(paste("Doing AMM-LMM for lambda,gamma = ", lambda, gamma))


          results_line <- tryCatch(
            {
              withTimeout(
                {
                  L <- laplacian(similarity = "G,s", trainset, N, sigma = 10)
                  w.amm <- alternating.mean.map(trainset, lambda = lambda, init = "LMM", L = L, gamma = gamma)
                  w.amm <- w.amm$theta # the algorithm returns a structure that contains also the number of step until termination
                },
                timeout = 3600
              )
              test.X <- as.matrix(testset[, -c(1, 2)])
              test.pred <- 1 / (1 + exp(-2 * test.X %*% w.amm))
              test.auc <- auc((testset$label + 1) / 2, test.pred)
              print(test.auc)
              results_line <<- paste(results_line, test.auc, sep = " , ")
            },
            TimeoutException = function(ex) {
              print("Timeout Occurred")
              results_line <<- paste(results_line, "timeout", sep = " , ")
              write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
              print(results_line)
              # print("Done this")
            },
            error = function(e) {
              print("Error Occurred")
              results_line <<- paste(results_line, "error", sep = " , ")
              write(results_line, file = clusterBagsMethodoutfile, append = TRUE)
              print(results_line)
              # print("Done this")
            }
          )
          print(results_line)
        }
      }

      write(results_line, file = clusterBagsMethodoutfile, append = TRUE)

      print(paste("Done: Name = ", name, "  Method = ", clusterBagsMethod, "  Split = ", splitnumber, "  Fold = ", foldnumber, sep = ""))
    }
  }
}
