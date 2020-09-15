from dominance_analysis import Dominance
import pandas as pd


def relativeImp():
    csv_file = "D:\\netSVG\\data\\functional\\train_data.csv"
    csv_data = pd.read_csv(csv_file, low_memory=False)
    train_data = pd.DataFrame(csv_data)
    train_data.columns = ["x0", "x1", "y"]
    dominance_regression = Dominance(data=train_data, target='y', objective=1)
    incr_variable_rsquare = dominance_regression.incremental_rsquare()
    dominance_regression.plot_incremental_rsquare()
    dominance_regression.dominance_stats()
    dominance_regression.dominance_level()
    imp = dominance_regression.percentage_incremental_r2
    return imp


def relativeImp_3d():
    csv_file = "D:\\netSVG\\data\\functional_3d\\train_data.csv"
    csv_data = pd.read_csv(csv_file, low_memory=False)
    train_data = pd.DataFrame(csv_data)
    train_data.columns = ["x0", "x1", "x2", "y"]
    dominance_regression = Dominance(data=train_data, target='y', objective=1)
    incr_variable_rsquare = dominance_regression.incremental_rsquare()
    dominance_regression.plot_incremental_rsquare()
    dominance_regression.dominance_stats()
    dominance_regression.dominance_level()
    imp = dominance_regression.percentage_incremental_r2
    return imp
