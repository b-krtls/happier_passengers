"""
:TODO:
Module Level Docstring
"""
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


class CorrelationCalculator:
    """A class to calculate correlations across variables

    Can be to calculate the correlations between different types of variables.
    All variable types are analysed among itself, with possibility of also 
    including the output features.
    Currently supported variable types: 
    - Ordinal
    - Nominal
    - Numerical
    Supported variable types can be accessed with 
        :attribute:'__allowed_vartypes'

    :return: CorrelationCalculator Class Instance with all information 
        encapsulated
    :rtype: instance of :class:'CorrelationCalculator'
    """

    __allowed_vartypes = ["ordinal", "nominal", "numerical"]

    def __init__(self,
                 data: pd.DataFrame,
                 var_indices:Iterable,
                 var_type: str,
                 include_output=False,
                 output_indices:Iterable=None,
                 ):
        """Initialize the CorrelationCalculator instance.

        By default, the last column is interpreted as output, if
        :include_output: = True and :output_indices: = None

        :param data: DataFrame that represents the dataset to calculate
            correlations across columns
        :type data: pd.DataFrame
        :param var_indices: A list/tuple/iterable that yields the index number
            of columns that are of variable type :var_type:
        :type var_indices: Iterable
        :param var_type: The type of variable for analysis. Currently,
            "ordinal", "nominal" and "numerical" is supported.
        :type var_type: str
        :param output_indices: An iterable that yields the variable indices 
            that belong to the output variables for data analysis, 
            defaults to None
        :type output_indices: Iterable, optional
        :param include_output: Boolean value to specify if calculations should 
            include output variables as specified in :output_indices:, 
            defaults to False
        :type include_output: bool, optional
        :raises ValueError: When the arguments to initialize the instance is 
            not allowed.
        """

        # Assign Default Values
        if var_type.lower() not in \
                self.__allowed_vartypes:
            raise ValueError("Specified var_type is not allowed")
        if output_indices == None:
            output_indices = [-1]

        if include_output:
            var_indices = var_indices + output_indices

        self.data = data
        self.data_cols = data.columns
        self.var_indices = var_indices
        self.var_type = var_type
        self.method: str = "" #
        self.matrix = ...

    def calculate(self):

        self.__calculation_switcher = {
            "numerical": self._numerical_correlation,
            "ordinal": self._ordinal_correlation,
            "nominal": self._nominal_correlation
        }
        self.__calculation_switcher[self.var_type]()
        return self.matrix

    def _numerical_correlation(self):

        self.method = "kendall"  # "pearson", "spearman" or "kendall"
        self.matrix = self.data[self.data_cols[self.var_indices]]\
            .corr(method=self.method)

    def _ordinal_correlation(self):
        self.method = "kendall"
        self.matrix = self.data[self.data_cols[self.var_indices]]\
            .corr(method=self.method)

    def _nominal_correlation(self):
        """Modified from https://stackoverflow.com/a/39266194
        Calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        self.method = "Bias-Corrected Cramer's V"

        pairs = []
        pairs_2_indices = dict()
        for ci, i in enumerate(self.var_indices):
            for cj, j in enumerate(self.var_indices):
                if ((i, j) not in pairs) and ((j, i) not in pairs):
                    pairs.append((i, j))
                    pairs_2_indices.update(
                        {
                            (i,j): (ci, cj),
                            (j,i): (cj, ci)
                        }
                    )

        size_ = len(self.var_indices)
        associations = np.empty((size_, size_))
        associations[:] = None
        for (i,j) in pairs: #enumerate([(1, 2)]):
            contingency_table = pd.crosstab(
                self.data[self.data_cols[i]],
                self.data[self.data_cols[j]],
                margins = False
            )
            chi2, p, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            phi2 = chi2/n
            r,k = contingency_table.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramersV = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

            ci, cj = pairs_2_indices[(i, j)]
            associations[ci, cj] = associations[cj, ci] = cramersV

        self.matrix = pd.DataFrame(
            data=associations,
            columns=self.data_cols[self.var_indices],
            index=self.data_cols[self.var_indices],
        )

    def illustrate(
            self,
            title = "",
            title_fontsize = 48,
            figsize=(20, 15),
            kwargs_seaborn_heatmap:dict=None,
            ):
        """
        Plot the matrix using a seaborn heatmap
        """

        # Default Value Assignments
        if not kwargs_seaborn_heatmap:
            kwargs_seaborn_heatmap = {
                "annot": True, 
                "annot_kws": {"size": 18}
            }
        if not title:
            title = f"{self.method.capitalize()} Correlation Matrix" \
                    f"\nfor {self.var_type.capitalize()} Variables"\
                    " in Dataset"

        sns.set(font_scale=3)
        fig = plt.figure(figsize=figsize)
        ax = sns.heatmap(self.matrix, **kwargs_seaborn_heatmap)
        ax.set_title(title, fontsize=title_fontsize)
        fig.add_axes(ax)
        sns.set(font_scale=1)
        return (fig, ax)

    def filter(self, column_name, threshold=1e-1) -> tuple:
        """Get the feature names of the original dataset and 
        their corresponding feature numbers, both as a list, where the 
        correlation value in :column_name: is smaller than :threshold: 
        (both in absolute value)

        Search through correlation matrix's specified column 
        and detect the rows where the absolute value of the 
        correlation value is smaller than the threshold.

        Note that since the correlation matrix is assembled to 
        be symmetric, the detected rows are also the columns of the 
        correlation matrix.

        :param column_name: The name of the column to be filtered
        :type column_name: str
        :param threshold: The threshold below which, the filter will 
            pick values, defaults to 1e-1
        :type threshold: float, optional
        :return: A tuple of values:
            [0] A list of index names, equal to the column names, 
            where the filter picks values from the class:'DataFrame'
            [1] A list of their corresponding numbers as the Nth feature
             of the original dataset, N starting from indexed from 0
        :rtype: tuple
        """

        matrix = self.matrix
        ls = self.data_cols.tolist()
        # redundant_cols = matrix[
        #     (-threshold< matrix[column_name]) & \
        #     (matrix[column_name] < threshold)
        #     ].index.tolist()  # Absolute value check
        redundant_cols = matrix[
            matrix[column_name].abs() < threshold
            ].index.tolist()  # Absolute value check
 
        feature_nums = [
            ls.index(c) for c in redundant_cols
        ]
        del matrix, ls
        return redundant_cols, feature_nums
