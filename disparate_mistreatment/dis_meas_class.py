from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import traceback
import numpy as np
from copy import deepcopy
import cvxpy as cp
import dccp
from dccp.problem import is_dccp
import dm_utils as ut


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs_to_cov_thresh, cons_type, w):
    """
    get the list of constraints to be fed to the minimizer

    cons_type == 0: means the whole combined misclassification constraint (without FNR or FPR)
    cons_type == 1: FPR constraint
    cons_type == 2: FNR constraint
    cons_type == 4: both FPR as well as FNR constraints

    sensitive_attrs_to_cov_thresh: is a dict like {s: {cov_type: val}}
    s is the sensitive attr
    cov_type is the covariance type. contains the covariance for all misclassifications, FPR and for FNR etc
    """

    constraints = []
    for attr in sensitive_attrs_to_cov_thresh.keys():

        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = ut.get_one_hot_encoding(attr_arr)

        if index_dict is None:  # binary attribute, in this case, the attr_arr_transformed is the same as the attr_arr

            s_val_to_total = {ct: {} for ct in [0, 1, 2]}  # constrain type -> sens_attr_val -> total number
            s_val_to_avg = {ct: {} for ct in [0, 1, 2]}
            cons_sum_dict = {ct: {} for ct in
                             [0, 1, 2]}  # sum of entities (females and males) in constraints are stored here

            for v in set(attr_arr):
                s_val_to_total[0][v] = sum(x_control_train[attr] == v)
                s_val_to_total[1][v] = sum(np.logical_and(x_control_train[attr] == v,
                                                          y_train == -1))  # FPR constraint so we only consider the ground truth negative dataset for computing the covariance
                s_val_to_total[2][v] = sum(np.logical_and(x_control_train[attr] == v, y_train == +1))

            for ct in [0, 1, 2]:
                s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][
                    1])  # N1/N in our formulation, differs from one constraint type to another
                s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0]  # N0/N

            for v in set(attr_arr):
                idx = x_control_train[attr] == v

                #################################################################
                # #DCCP constraints
                dist_bound_prod = cp.multiply(y_train[idx], x_train[idx] @ w)  # y.f(x)

                cons_sum_dict[0][v] = cp.sum(cp.minimum(0, dist_bound_prod)) * (
                            s_val_to_avg[0][v] / len(x_train))  # avg misclassification distance from boundary
                cons_sum_dict[1][v] = cp.sum(
                    cp.minimum(0, cp.multiply((1 - y_train[idx]) / 2.0, dist_bound_prod))) * (
                                                  s_val_to_avg[1][v] / sum(
                                              y_train == -1))  # avg false positive distance from boundary (only operates on the ground truth neg dataset)
                cons_sum_dict[2][v] = cp.sum(
                    cp.minimum(0, cp.multiply((1 + y_train[idx]) / 2.0, dist_bound_prod))) * (
                                                  s_val_to_avg[2][v] / sum(
                                              y_train == +1))  # avg false negative distance from boundary
                #################################################################

            if cons_type == 4:
                cts = [1, 2]
            elif cons_type in [0, 1, 2]:
                cts = [cons_type]

            else:
                raise Exception("Invalid constraint type")
            #################################################################
            # DCCP constraints
            for ct in cts:
                thresh = abs(sensitive_attrs_to_cov_thresh[attr][ct][1] - sensitive_attrs_to_cov_thresh[attr][ct][0])
                constraints.append(cons_sum_dict[ct][1] <= cons_sum_dict[ct][0] + thresh)
                constraints.append(cons_sum_dict[ct][1] >= cons_sum_dict[ct][0] - thresh)
            #################################################################
        else:  # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
            # need to fill up this part
            raise Exception("Fill the constraint code for categorical sensitive features... Exiting...")
    return constraints


class DispMistreatmentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 x_control,
                 cons_type=None,
                 tau=5.,
                 mu=1.2,
                 loss_function="logreg",
                 EPS=1e-6):
        self.sensitive_attrs = [i for i in x_control.keys()]
        self.x_control = x_control

        sensitive_attrs_to_cov_thresh = {}
        for attr in self.sensitive_attrs:
            sensitive_attrs_to_cov_thresh[attr] = \
                {0: {0: 0, 1: 0},
                 1: {0: 0, 1: 0},
                 2: {0: 0, 1: 0}
                 }  # zero covariance threshold,
                    # means try to get the fairest solution
        if cons_type is None:
            self.cons_params = None
        else:
            self.cons_params = {"cons_type": cons_type,
                                "tau": tau,
                                "mu": mu,
                                "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

        self.loss_function = loss_function
        self.EPS = EPS

    def fit(self, X, y=None):
        max_iters = 500  # for the convex program
        max_iter_dccp = 100  # for the dccp algo

        num_points, num_features = X.shape
        w = cp.Variable(num_features)  # this is the weight vector

        # initialize a random value of w
        np.random.seed(112233)
        w.value = np.random.rand(X.shape[1])

        if self.cons_params is None:  # just train a simple classifier, no fairness constraints
            constraints = []
        else:
            w = np.array(w.value).flatten()  # flatten converts it to a 1d array
            self.w = w
            constraints = get_constraint_list_cov(X, y,
                                                  self.x_control,
                                                  self.cons_params["sensitive_attrs_to_cov_thresh"],
                                                  self.cons_params["cons_type"],
                                                  self.w)

        if self.loss_function == "logreg":
            # constructing the logistic loss problem
            loss = cp.sum(
                cp.logistic(cp.multiply(-y, X @ w))) / num_points  # we are converting y to a diagonal matrix for consistent

        # sometimes, its a good idea to give a starting point to the constrained solver
        # this starting point for us is the solution to the unconstrained optimization problem
        # another option of starting point could be any feasible solution
        if self.cons_params is not None:
            if self.cons_params.get("take_initial_sol") is None:  # true by default
                take_initial_sol = True
            elif self.cons_params["take_initial_sol"] == False:
                take_initial_sol = False

            if take_initial_sol == True:  # get the initial solution
                p = cp.Problem(cp.Minimize(loss), [])
                p.solve()

        # construct the cvxpy problem
        prob = cp.Problem(cp.Minimize(loss), constraints)

        try:
            tau, mu = 0.005, 1.2  # default dccp parameters, need to be varied per dataset
            if self.cons_params is not None:  # in case we passed these parameters as a part of dccp constraints
                if self.cons_params.get("tau") is not None: tau = self.cons_params["tau"]
                if self.cons_params.get("mu") is not None: mu = self.cons_params["mu"]

            try:
                prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
                           solver=cp.ECOS, verbose=False,
                           feastol=self.EPS, abstol=self.EPS,
                           reltol=self.EPS, feastol_inacc=self.EPS, abstol_inacc=self.EPS,
                           reltol_inacc=self.EPS,
                           max_iters=max_iters, max_iter=max_iter_dccp)
            except cp.SolverError:
                prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
                           solver=cp.MOSEK, verbose=False)
            print("Optimization done, problem status:", prob.status)
            assert (prob.status == "Converged" or prob.status == "optimal")
        except:
            traceback.print_exc()
            sys.stdout.flush()
            sys.exit(1)

        # check that the fairness constraint is satisfied
        for f_c in constraints:
            assert (f_c.value() == True)    # can comment this out if the solver fails too often,
            # but make sure that the constraints are satisfied empirically.
            # alternatively, consider increasing tau parameter
            pass

        w = np.array(w.value).flatten()  # flatten converts it to a 1d array

        self.w = w

    def predict(self, X, y=None):
        assert (len(self.sensitive_attrs) == 1)  # ensure that we have just one sensitive attribute
        s_attr = self.sensitive_attrs[0]  # for now, lets compute the accuracy for just one sensitive attr

        # compute distance from boundary
        distances_boundary = self.get_distance_boundary(self.w, X, self.x_control[s_attr])

        # compute the class labels
        y_pred = np.sign(distances_boundary)
        return y_pred

    def get_fpr_fnr_sensitive_features(self, X, y, y_pred, verbose=False):
        # we will make some changes to x_control in this function,
        # so make a copy in order to preserve the origianl referenced object
        x_control_internal = deepcopy(self.x_control)

        s_attr_to_fp_fn = {}

        for s in self.sensitive_attrs:
            s_attr_to_fp_fn[s] = {}
            s_attr_vals = x_control_internal[s]
            if verbose == True:
                print('Accuracy: {:.3f}'.format(self.get_score(X, y_pred, y)))
                print("||  s  || FPR. || FNR. ||")
            for s_val in sorted(list(set(s_attr_vals))):
                s_attr_to_fp_fn[s][s_val] = {}
                y_true_local = y[s_attr_vals == s_val]
                y_pred_local = y_pred[s_attr_vals == s_val]

                acc = float(sum(y_true_local == y_pred_local)) / len(y_true_local)

                fp = sum(np.logical_and(y_true_local == -1.0,
                                        y_pred_local == +1.0))  # something which is -ve but is misclassified as +ve
                fn = sum(np.logical_and(y_true_local == +1.0,
                                        y_pred_local == -1.0))  # something which is +ve but is misclassified as -ve
                tp = sum(np.logical_and(y_true_local == +1.0,
                                        y_pred_local == +1.0))  # something which is +ve AND is correctly classified as +ve
                tn = sum(np.logical_and(y_true_local == -1.0,
                                        y_pred_local == -1.0))  # something which is -ve AND is correctly classified as -ve

                all_neg = sum(y_true_local == -1.0)
                all_pos = sum(y_true_local == +1.0)

                fpr = float(fp) / float(fp + tn)
                fnr = float(fn) / float(fn + tp)
                tpr = float(tp) / float(tp + fn)
                tnr = float(tn) / float(tn + fp)

                s_attr_to_fp_fn[s][s_val]["fp"] = fp
                s_attr_to_fp_fn[s][s_val]["fn"] = fn
                s_attr_to_fp_fn[s][s_val]["fpr"] = fpr
                s_attr_to_fp_fn[s][s_val]["fnr"] = fnr

                s_attr_to_fp_fn[s][s_val]["acc"] = (tp + tn) / (tp + tn + fp + fn)
                if verbose == True:
                    if isinstance(s_val, float):  # print the int value of the sensitive attr val
                        s_val = int(s_val)
                    print('||  {}  || {:.2f} || {:.2f} ||'.format(s_val, fpr, fnr))

            return s_attr_to_fp_fn


    def get_score(self, X, y_pred, y):
        correct_answers = self.get_amt_correct_ans(X, y_pred, y)
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy


    @staticmethod
    def get_distance_boundary(w, x, s_attr_arr):
        """
            if we have boundaries per group, then use those separate boundaries for each sensitive group
            else, use the same weight vector for everything
        """

        distances_boundary = np.zeros(x.shape[0])
        if isinstance(w, dict):  # if we have separate weight vectors per group
            for k in w.keys():  # for each w corresponding to each sensitive group
                d = np.dot(x, w[k])
                distances_boundary[s_attr_arr == k] = d[
                    s_attr_arr == k]  # set this distance only for people with this sensitive attr val
        else:  # we just learn one w for everyone else
            distances_boundary = np.dot(x, w)
        return distances_boundary


    @staticmethod
    def get_amt_correct_ans(X, y_pred, y):
        correct_answers = (y_pred == y).astype(int)  # will have 1 when the prediction and the actual label match
        return correct_answers


