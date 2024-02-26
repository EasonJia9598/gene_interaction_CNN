# nolint
#' Process the output of `EstimateCCM`
#'
#' Calculate the convergence rate and standard errors.
#'
#' @param aE a list returned by `EstimateCCM()`.
#' @return a list with following elements:
#' \itemize{
#' \item rate: rate of converge. If value is too large (e.g. >300), consider increasing the tuning parameter \eqn{\lambda}.
#' \item hessianSE: estimated standard errors using Hessian matrix.
#'
#' }
#'
#' @examples
#' set.seed(123)
#' t <- rtree(100)
#' d <- TreeToDend(t)
#'
#' # setting random parameters for a pair without interaction
#' n <- 2
#' alpha <- runif(n, -0.1, 0.1)
#' B <- matrix(0, n, n)
#' diag(B) <- runif(n, -0.1, 0.1)
#' B[1,2] <- B[2,1] <- 0 # independent pair
#'
#' simDF <- SimulateProfiles(t, alpha, B)
#' ProfilePlot(simDF, d) # plot the profiles
#' aE <- EstimateCCM(profiles = simDF, phytree=t)
#' estSE <- ProcessAE(aE)$hessianSE
#' # testing if there is significant interaction
#' # p value for Ha: \eqn{\beta != 0}
#' sigScore <- aE$nlm.par[5] / estSE[5]
#' print(2*(1 - pnorm(abs(sigScore))))
#'
#' # simulate a pair with interaction
#' B[1,2]<-B[2,1] <- 0.5 # set an interaction between genes
#' simDF <- SimulateProfiles(t, alpha, B)
#' ProfilePlot(simDF, d)
#' aE <- EstimateCCM(profiles = simDF, phytree=t)
#' estSE <- ProcessAE(aE)$hessianSE
#' # testing if there is significant interaction
#' # p value for Ha: \eqn{\beta != 0}
#' sigScore <- aE$nlm.par[5] / estSE[5]
#' print(2*(1 - pnorm(abs(sigScore))))
#'
#' @export

ProcessAE = function(aE){ # nolint
    if ( inherits( try( hmatrix <- aE$nlm.hessian, silent=TRUE),  "try-error")){ # nolint
        cvrate = NA # nolint    
        hessianSE = NA
    } else {

        eigh = eigen(hmatrix)
        lambda1 = eigh$values[1]
        lambda0 = eigh$values[length(eigh$values)]
        cvrate = lambda1/lambda0
        hessianSE = getSE(hmatrix)
    }
    return(list(rate=cvrate, hessianSE = hessianSE))
}


getSE <- function(h)
{

    if (any(diag(h) == 0)) {
        warning("Maximum likelihood estimation seems not converging correctly. Consider using different initial values and tuning parameters.")
        se <- rep(NaN, nrow(h))
    }
    else {
        se <- sqrt(diag(solve(h)))
    }
    return(se)
}


