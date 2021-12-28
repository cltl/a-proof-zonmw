"""
create patient recovery profiles
- select interesting profiles from file 004
- gather ADM levels over time from selected profiles
"""
import pandas as pd
import matplotlib.pyplot as plt

def select_profile(filepath):
    """
    select profiles from the 004 umcu file 
    patients that have been in the hospital for a minimum of 30 days and have been discharged to go home
    """
    dataframe = pd.read_csv(filepath, sep = ';')
    
    #df2 = dataframe.loc[(((dataframe['BestemmingID'] == 'H') & (dataframe['Opnamedatum'] == '01-01-2021') & (dataframe['Ontslagdatum'] == '30-01-2021')))]
    #df2 = dataframe.loc[(((dataframe['BestemmingID'] == 'H') & (dataframe['Opnamedatum'] == '01-12-2020') & (dataframe['Ontslagdatum'] == '17-01-2021')))]
    #df2 = dataframe.loc[(((dataframe['BestemmingID'] == 'O') & (dataframe['Opnamedatum'] == '01-02-2021') & (dataframe['Ontslagdatum'] == '07-02-2021')))]
    df2 = dataframe.loc[(((dataframe['BestemmingID'] == 'O') & (dataframe['Opnamedatum'] == '01-03-2021') & (dataframe['Ontslagdatum'] == '19-03-2021')))]
    
    pseudo_bsn = df2['Pseudoniem BSN'].tolist()
    
    return(pseudo_bsn)

def get_recovery_pattern(filepath, psuedo_bsn):
    """
    :param pseudo_bsn: lst
    """
    
    df_levels = pd.read_csv(filepath, sep = ';')
    
    #select the rows of the first patient in the list
    df_patient = df_levels.loc[df_levels['Pseudoniem BSN'] == psuedo_bsn[0]]
    
    #create new column where dates are in datetime type
    df_patient['datum_datetime'] = pd.to_datetime(df_patient['datum'], format="%d-%m-%Y")
    
    #sort dataframe on date
    df_patient = df_patient.sort_values('datum_datetime', ascending=True)
    
    plt.plot(df_patient['datum_datetime'], df_patient['ADM_lvl'], '*')
    plt.xticks(rotation = 'vertical')
    plt.show()
    
def main():
    
    # the list of patients is currently 1
    list_of_patients = select_profile("../data/umcu/APROOF_UMCU_20211126_004.csv")
    # get a plot
    get_recovery_pattern("../data/umcu/APROOF_UMCU_20211126_008.csv", list_of_patients)
    
if __name__ == "__main__":
    main()
    

    
    
    
    
    
    
    
    
    

                            
