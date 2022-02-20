# Happier Passengers

This project is about analyzing the [Kaggle Dataset](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction), which contains Airline Passenger Satisfaction data. It contains a multitude of different features that may be used to analyze which parameters influence the satisfaction of airlines.

The project mainly consists of the [Jupyter File](data_analysis.ipynb) which contains the executed steps to analyze and model the data and draw conclusions. There is an auxiliary [Python File](correlationcalc.py) which implements correlation calculation of variables. It is semi-generic and may be used in other projects with similar intent, but in its current version, it is useful for calculating correlations of alike-variables (the c/ross-correlation of ordinal, numerical, nominal variables cannot be addressed.)

## DataSet

For this dataset, the default properties are (specified by the owner of dataset) as such:

- Gender: Gender of the passengers (Female, Male)
- Customer Type: The customer type (Loyal customer, disloyal customer)
- Age: The actual age of the passengers
- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
- Flight distance: The flight distance of this journey
- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable; 1-5)
- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
- Ease of Online booking: Satisfaction level of online booking
- Gate location: Satisfaction level of Gate location
- Food and drink: Satisfaction level of Food and drink
- Online boarding: Satisfaction level of online boarding
- Seat comfort: Satisfaction level of Seat comfort
- Inflight entertainment: Satisfaction level of inflight entertainment
- On-board service: Satisfaction level of On-board service
- Leg room service: Satisfaction level of Leg room service
- Baggage handling: Satisfaction level of baggage handling
- Check-in service: Satisfaction level of Check-in service
- Inflight service: Satisfaction level of inflight service
- Cleanliness: Satisfaction level of Cleanliness
- Departure Delay in Minutes: Minutes delayed when departure
- Arrival Delay in Minutes: Minutes delayed when Arrival
- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)

Satisfaction levels of different services are encoded in likert-scale by default.

## TODO

- [] Implement a more intelligent CorrelationCalculator class, which quantifies the correlation across any two type of variable. (e.g. quantify correlation between a numerical variable and a categorical variable)