

# Hotel Cancellation Analysis

**Group 2:**  Thomas Dunn, Isaac Gish, Shannon Lloyd, Jason Miller, Tait Ralston

------
## Table of Contents

- [Purpose](#purpose)
- [Usage](#usage)
- [Data](#data)
- [Process](#process)
- [Visualizations](#visualizations)
- [Analysis](#analysis)
- [Contributors](#contributors)
------

## Purpose

Use machine learning (ML) with the other technologies we’ve learned to analyze different attributes of customers’ hotel booking details to create a model that can predict whether or not the booking will be canceled.

## Usage

All files for retrieving, cleaning, and modeling our data is in: 

[HotelCancellationAnalysis.ipynb](https://github.com/tomwildun/Project4/blob/main/HotelCancellationAnalysis.ipynb) 

This was run locally in JupyterLab with Pandas, Scikit-Learn, PySpark , and TensorFlow  installed on a Mac system.

The slide show is available at: 

[Group 2 Project 4 Presentation](https://docs.google.com/presentation/d/1iR5Na8Q63F8MP9ENuOVx30hnzs8L5aWCDCOfFt5XU60/edit#slide=id.g15207a17831_0_143)

The Tableau Viz is available at:

[Hotel Cancelation Analysis](https://public.tableau.com/app/profile/shannon.lloyd/viz/HotelCancelationAnalysis_17005441395550/Welcome)



## Data

The dataset includes data from the INN Hotels Group, which has a chain of hotels in Portugal. We used the dataset available from Mariyam Al Shatta on Kaggle. The data was already very clean, appeared well suited for machine learning, and is public domain.  

[INN Hotels Group](https://www.kaggle.com/datasets/mariyamalshatta/inn-hotels-group/data) 

## Process

1.  Imported CSV file with PySpark.  

   ![img](https://lh7-us.googleusercontent.com/IdyWV_XC7D6AbjhPIMOcfB1LagoD8ipwKwllBFdbStBRiXJH6PuvOZ7II9Ey_4m8y2CZHC9B-CDps_KG4em9Vk3GxLx0mgwY5GFPxQExwEFhYYi0Gk97QFtUfvJTUrU28TCBS7f4GVaBRKUiy25bQTWJ0g=s2048)

2. Utilized Python Pandas the data is cleaned, normalized, and standardized prior to modeling.

   - Data Processing - Generated Dummy variables to change data into numerical form for processing.![img](https://lh7-us.googleusercontent.com/iBXSauKvbsoyhNdjcPhc9ds2FT2fq5EmAOlIXsez1Al91jP4kpB_DzXbTRfJhlTx7qIH1JZ0Oo_PBkavgD45bkMuerFGLBrPFAtoTbUCtmDXm8in7zXKvb11T-CMMlAcYT3ducstQhySZi3aIy-gTt_uhQ=s2048)

   - Data Processing - Encoding variables to boolean values![img](https://lh7-us.googleusercontent.com/9bxQbEPsrMlhdUoecixWQK8MRNB58Agq4WbyWVa80DEhRJsXRApfl0XoFr4ccSDCkXr3ywsBWWJwYxPBMv7-oWFTz-YGOtlvrKHxrrH2sE1tGoA1HvtBOiRFRRAgs0omaCLmZ2Y9oiXSIRfTqTxgQuHGrw=s2048)

     

3. We used a TensorFlow model but after 2 Epochs we achieved 100% indicating overfitting of the data. We set out to try two other models, Logistic Regression & Random Forest to find the best fit. Random Forest provided us with the best results. 

   - Neural Network TensorFlow Model![img](https://lh7-us.googleusercontent.com/lCp8kROxb6Xd92a0Ql2pgIdRrxZ4-8BWgcH5nVmHcAfD29_EM56iPtJirdvzNW9NLa358B1qMosph1z3I0dc4Fz2pW65jIewZqaUN_eWzuG8ydqPwllp2kasrCCJWu_8klnWBxoqWXyswyHsFIiD73itRg=s2048)
   - Logistic Regression Model![img](https://lh7-us.googleusercontent.com/3I1ISWfgbDagGmYbdYjBzT6ZDzVvAIBpAGTj7ZVVW_UvGOcOZLLBJHpew1uKaFUCiUWhJCSyNcuQTs-Q85-BzUseB9K_9OT6tjClSGUBNxRNf9wzLktkUnD5iOk28nmgVGGPerA2T9l21lj8oAP1lck3Bw=s2048)
   - Random Forest Model![img](https://lh7-us.googleusercontent.com/G5hpf1SsSNUKg3TUIhgddUUjXcJEfJm0c0nP0OaFWdvymG9b-czv0Vj3Du4ZdBf2uZHuQDE8uG5-9GRyQKMSq88oLSZU96esO_Iz1EZE4HROtU3C8tfGU6N7hW_xsLR8B8cH9Pb8h-qFiFMb71YRdNddlA=s2048)

   ![img](https://lh7-us.googleusercontent.com/4H-oe5G9fG_9XoURHQcZ0x-2Iw_Qb8IIFudmXmpkl3RgNraRYyBOctDsHG9JOy13seFh47hJu_NKI6hfMdvL2axErLeraA4cwwEN4gE0VgjqZmp_s-pKnEGEYjf55thj5PctOeaX9bKL0ZAKWTjhTqcIYQ=s2048)

## Visualizations

1. Reservation Features

   - Lead Time:  Advance bookings with a long lead time have higher rates of cancelation.

   - Price per Room:  As cost per room rises so does cancellations.

     ![img](https://github.com/tomwildun/Project4/blob/main/Resources/ReservationFeatures.png?raw=true)

2. Customer Requests

   - Reservation has Children: Increase in cancelations.
   - Special Requests and Parking Needed: Decrease in cancelations.![img](https://github.com/tomwildun/Project4/blob/main/Resources/CustomerRequests.png?raw=true)

3. Meal Plans

   - Meal plan 2: Only meal plan with a significant increase in cancellations. ![img](https://github.com/tomwildun/Project4/blob/main/Resources/MealPlan.png?raw=true)

4. Customer Loyalty

   - Repeat Guest: Very low cancelation rates
   - Previous Stays: Very low cancelation rates regardless if the cancelled previously ![img](https://github.com/tomwildun/Project4/blob/main/Resources/CustomerLoyalty.png?raw=true)

5. Booking Type

   - Online: has a significantly higher rate of cancellations compared to other booking types.![img](https://github.com/tomwildun/Project4/blob/main/Resources/MarketSegment.png?raw=true)

6. Monthly Bookings

   - Highest Cancellation Rate: July
   - Lowest Cancellation Rate: December ![](https://github.com/tomwildun/Project4/blob/main/Resources/MonthlyBookings.png?raw=true)

   

## Analysis

Through our data modeling we learned that it is important to pick the right model for your data. We achieved very different results with various data models. TensorFlow was overfitting so it was not a good fit. Logistic Regression was good but Random Forest was a great fit for our data. Using the Random Forest Model we were able to achieve 90% accuracy. We also learned that Lead Time and Price per Room were the variables that had the great predictors  of whether a booking was going to be cancelled.

This led us to start our visualizations with Lead Time and Average Price per Room. We could clearly see that these feature had a visible correlation with cancellation rates. As the Lead Time to bookings grew so did the rate of cancellations. Also as Price per Room increased so did the cancellations.

We also looked at the less important features and were able to see many areas that would be worthwhile to research further.  Customers with children have higher cancellations whereas customers that have special requests or require parking actually have lower cancelations. Some possibilities to consider may be adults only, encourage customizable reservations that accommodate special requests, and make sure that a parking option is available. Although we do not know the specifics of Meal plan 2 the increase in cancellations associated with it warrants further research. Repeat customers have very low cancellation rates so offering loyalty programs may be something to consider. Further research is needed to find out why there is such a stark difference between the cancellation rates in July vs December. Although these are not the most important features they are still something to consider.

In conclusion, data modeling and visualizations proved to be effective tools to gain insight into hotel cancellations.

## Contributors

- Thomas Dunn [GitHub](https://github.com/tomwildun) 
- Isaac Gish [GitHub](https://github.com/isaac-gish) | [LinkedIn](https://www.linkedin.com/in/isaac-gish-b7074b261/)
- Shannon Lloyd [GitHub](https://github.com/sunshinebearlloyd) | [LinkedIn](https://www.linkedin.com/in/shannon-lloyd-132952279/)
- Jason Miller 
- Tait Ralston [GitHub](https://github.com/tralsto) | [LinkedIn](https://www.linkedin.com/in/taitralston/)