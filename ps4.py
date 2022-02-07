import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import interp1d

#brooke pulling
#time spent: 12 hours
#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper-mean)/st.norm.ppf(.95)

def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)

def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into a numpy array

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year','Lower','Upper']
    return (df.Year.to_numpy(),df.Lower.to_numpy(),df.Upper.to_numpy())

###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    years,low_vals,hi_vals = load_slc_data()
    
    array = []
    #build the array
    for i in range(2020,2101): 
        if i in years:
            index = int((i - 2020)/10)
            year = i
            low_val = low_vals[index]
            hi_val = hi_vals[index]
            mean = (low_val + hi_val)/2
            std = calculate_std(hi_val, mean)
            array.append([year, mean, low_val, hi_val, std])
        else:
            year = i
            low_val = interp(i, years, low_vals)
            hi_val = interp(i, years, hi_vals)
            mean = (low_val + hi_val)/2
            std = calculate_std(hi_val, mean)
            array.append([year, mean, low_val, hi_val, std])
    
    #build the plot 
    if show_plot == True:
        years = []
        mean_list = []
        updated_low_vals = []
        updated_hi_vals = []
        for i in range(81):
            years.append(array[i][0])
            mean_list.append(array[i][1])
            updated_low_vals.append(array[i][2])
            updated_hi_vals.append(array[i][3])
        years = np.linspace(2020, 2100, 81)
        plt.plot(years, mean_list, label="Mean")
        plt.plot(years, updated_low_vals, linestyle="dashed", label="Lower")
        plt.plot(years, updated_hi_vals, linestyle="dashed", label="Upper")
        plt.xlabel('Year')
        plt.ylabel('Projected annual mean water level (ft)')
        plt.legend(loc="upper left")
        plt.axis([2020,2100, 3.5, 10.3])
        plt.show()
        
    return np.array(array)

def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    std = data[year-2020][4]
    mean = data[year-2020][1]
    array = []
    for sim_val in range(num):
        array.append(np.random.normal(mean, std))
    return np.array(array)


def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    year = 2020
    plt.xlabel('Year')
    plt.ylabel('relative water level change(ft)')
    while year < 2100:
        plot_points = []
        for i in range(500):
            plot_points.append(year)
        year += 1
        years = np.linspace(2020, 2100, 81)
        plt.scatter(plot_points, simulate_year(data, year, 500),s=.01,c="darkgrey")
    plt.title('expected results')
    plt.plot(years, data[:,1], label="Mean")
    plt.plot(years, data[:,2], linestyle="dashed", label="Lower")
    plt.plot(years, data[:,3], linestyle="dashed", label="Upper")
    plt.axis([2020,2100, 3.5, 10.3])
    plt.legend(loc="upper left")
    plt.show()
    

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""

    water_levels = []
    year = 2020
    while 2020 <= year <= 2100:
        water_levels.append((simulate_year(predicted_sea_level_rise(), year, 1)))
        year += 1
    return water_levels

def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_list = []
    #this dictionary maps slr value to damage
    #dic = {}
    #[:,0] means iterate through all rows, zeroth column
    #iterate thru the index of the array
    #for i in range(water_level_loss_no_prevention.shape[0]):
        #[slr:1] row slr 
        # slr = water_level_loss_no_prevention[i:1] (ie. go to the ith index and get the zero column)
        #dic[water_level_loss_no_prevention[i, 0]] = water_level_loss_no_prevention[i, 1]
    interp_fuction = interp1d(water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1], fill_value = 'extrapolate')
    for i in range(len(water_level_list)):
        if water_level_list[i] <= 5:
            damage_list.append(0)
        elif water_level_list[i] >= 10:
            damage_list.append(house_value)
        # elif water_level_list[i][0] in dic:
        #     val = (dic[water_level_list[i]]*house_value)/100000
        #     damage_list.append(val)
        #need to interpolate
        else:
            #create an interp function by giving it all the slr vals and all the damage vals as data
            val = (interp_fuction(water_level_list[i])*house_value)/100000
            damage_list.append(val)
    
    return damage_list
            
            
def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
        
    damage_list = []
    prevention = False
    for water_level in water_level_list:
        if water_level <= 5:
            damage_list.append(0)
        elif water_level >= 10:
            damage_list.append(house_value/1000)
        #for the case where water level is between 5 and 10ft
        else:
            if prevention == True:
                percent_damage = interp(water_level, water_level_loss_with_prevention[:,0], water_level_loss_with_prevention[:,1])
                damage_list.append(house_value*percent_damage/100000)
            else:
                percent_damage = interp(water_level, water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1])
                if house_value*percent_damage/100 >= cost_threshold:
                    prevention = True
                damage_list.append(house_value*percent_damage/100000)
    return damage_list


def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    return repair_only(water_level_list, water_level_loss_with_prevention)
    


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    
    years = list(range(2020, 2101))

    #repair val is a list 
    #we have 500 repair_val lists, and we want to average the first, second, etc. trial across  500 lists
    #so make a 2d array where the rows are the repair lists 
    #then average the rows and scatter those values
    
    #creating an array
    repair_list = []
    wait_list = []
    prep_list = []
    
    test_range = 500
    
    
    
    for trial in range(test_range):
        water_est = water_level_est(data)
     
        repair_val = repair_only(water_est, water_level_loss_no_prevention, house_value)
        repair_list.append(repair_val)
    
        wait_val = wait_a_bit(water_est, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
        wait_list.append(wait_val)
        
        prep_val = prepare_immediately(water_est, water_level_loss_with_prevention, house_value)
        prep_list.append(prep_val)
        print(trial)
        
    year_2d = years*test_range

    repair_mean = np.mean(repair_list, axis = 0)
    wait_mean = np.mean(wait_list, axis = 0)
    prep_mean = np.mean(prep_list, axis = 0)

    plt.scatter(year_2d,repair_list, s = .1, c = "red")
    plt.scatter(year_2d, wait_list, s = .1, c = "blue")
    plt.scatter(year_2d, prep_list, s = .1, c = "green")
    plt.plot(years, repair_mean, label = 'repair only scenario', c = 'red')
    plt.plot(years, wait_mean, label = 'wait a bit scenario',c = "blue")
    plt.plot(years, prep_mean, label = 'prepare immediately scenario', c = "green")
   
    plt.legend(loc="upper left")
    plt.xlabel('year')
    plt.ylabel('estimated damage cost ($K)')
    plt.title('annual average damage cost')
    plt.axis([2020,2100, 0, 400])
    plt.show()


if __name__ == '__main__':
    data = predicted_sea_level_rise(show_plot = True)
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    # plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
