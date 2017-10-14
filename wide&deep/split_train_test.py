from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    df = pd.read_csv("./fin_data.csv")
    label = df.pop('label')
    df.insert(0, 'label', label)
    df_train =  df[df.columns[1:114]]
    df_target = df['label']
    train_X, test_X, train_y, test_y = train_test_split(
        df_train, df_target, test_size=0.3)
    train_X.insert(0,'label',train_y) 
    test_X.insert(0,'label',test_y)
    '''
    train_X.to_csv('train_data.csv',index=False)
    test_X.to_csv('test_data.csv',index=False)
    '''
    print("Success")
    
if __name__=="__main__":
    main()
