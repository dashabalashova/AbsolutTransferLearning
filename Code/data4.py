import pandas as pd

df = pd.read_csv('../Data/1S78_B_Mascotte_feat.txt', sep='\t', header=1)
df = df[['Slide', 'Energy']].sort_values(by='Energy')
df0 = df.tail(int(df.shape[0]*0.5))
df0['AgClass'] = 0
df0['AASeq'] = df0.apply(lambda x: x.Slide[:10], axis=1)
df1 = df.head(int(df.shape[0]*0.2))
df1['AgClass'] = 1
df1['AASeq'] = df1.apply(lambda x: x.Slide[:10], axis=1)
df2 = df0[['AASeq', 'AgClass']].append(df1[['AASeq', 'AgClass']]).sample(frac=1, random_state=0).reset_index(drop=True)
df2.to_csv('../Data/data_Absolut4.tsv', sep='\t', header=True)