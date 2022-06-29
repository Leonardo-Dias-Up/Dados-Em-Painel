import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.formula.api as sms
from statsmodels.formula.api import ols
import statsmodels.api as sm
from linearmodels import PanelOLS #biblioteca para dados de painel fixo
from linearmodels import PooledOLS #biblioteca de regressão modelo pooled
from linearmodels import RandomEffects #biblioteca de efeitos randomicos
import numpy.linalg as la
from scipy import stats
import scipy as sp
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kurtosis, skew, normaltest, jarque_bera
from linearmodels import PanelOLS
from linearmodels.panel import generate_panel_data
import math
import wbdata
import datetime 

# =============================================================================
# Base de Dados
# =============================================================================
df = pd.read_excel(r"D:\07. UFOP\Econometria I\Trabalho Final\Dados em Painel\03. Base de Dados.xlsx")
df.head()

# Time Index
df = df.set_index(["country","date"]).stack().unstack()
df.head()
df.tail()   
# variável dependente/endógena
y1 = df.iloc[:,0]
y1.head()

# Elabora um histograma
# plt.hist(y1, color='red', bins=15)
# plt.title('Histograma da variável resposta')
# plt.show()

# dependente
endog = y1
endog.head()

# independentes
exog = df.iloc[:,1:]
exog.columns

# =============================================================================
# MODELO OLS (Ordinary Least Square) 
# =============================================================================
# Mínimos Quadrados Ordinários
# Com constante alfa
#results = sm.OLS(endog, sm.add_constant(exog)).fit()
# Sem Contante alfa
results = sm.OLS(endog, exog).fit()
print(results.summary())

def Diagnostico_Residuos():
    print("Média do Erro", round(results.resid.mean(),4))
    # Desvio Padrão do Erro
    print("Desvio Padrão do Erro", round(results.resid.std(),4))
    # Skewness: assimetria da distribuição. Quando mais próximo de 0 mais “perfeita” é a assimetria; para valores > 0 existe uma assimetria positiva, e negativa para valores < 0. Também é possível calcular a skewness com um método do pandas chamado skew().
    print("Skewness", round(skew(results.resid),4))
    # Kurtosis: a curtose está associada ao achatamento da distribuição. A curtose de uma distribuição normal é 3. Para valores > 3 a distribuição é mais “alta” que a distribuição normal e para valores < 3 mais “achatada”. Também é possível calcular a curtose com um método do pandas chamado kurtosis(), porém o pandas calcula utilizando a definição de Fisher, que considera a curtose da normal igual a 0. Ou seja, esse resultado será expresso como um “excesso” de curtose, que é o valor do saldo após a subtração por 3. Acima calculamos das duas formas.
    print("Kurtosis", round(kurtosis(results.resid),4))
    # Teste Jarque-Bera e normaltest: ambos os testes tem como hipótese nula a normalidade. Sendo assim, para valores de p < 0,05 a normalidade é rejeitada.
    print("Jarque Bera", round(jarque_bera(results.resid)[0],4))
    print(" -> P(Jarque Bera)", round(jarque_bera(results.resid)[1],4))
    # Teste Normaltest: o testes tem como hipótese nula a normalidade. Sendo assim, para valores de p < 0,05 a normalidade é rejeitada.
    print("Normalidade", round(normaltest(results.resid)[0],4))
    print(" -> P(Normalidade)", round(normaltest(results.resid)[1],4))

Diagnostico_Residuos()

# Teste homocedasticidade - Goldfeld-quandt
def Teste_Goldfeld_Quandt():
    test_calculado = round(het_goldfeldquandt(results.resid,
                                    results.model.exog)[0],3)
    
    p_teste = het_goldfeldquandt(results.resid,
                     results.model.exog)[1]
    
    if p_teste>0.05:
        print("A hipotese nula da Homocedasticidade é aceita!")
        print("O t calculado para o Teste GQ é:",test_calculado)
        print("A P(t calculado) para o Teste GQ é:",p_teste)
    else:
        print("A hipotese nula é rejeitada! \nHá hetorocedasticidade!")
    
Teste_Goldfeld_Quandt()

# Teste VIF - 
def Teste_VIF():
    x = exog
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    print(vif_data)
    print('\n*obs: Valores Maiores que 7 indicam multicolinearidade')
    if results.condition_number > 1000:
        print('\nresults.condition_number: %s \
              \nThis might indicate that there are strong multicollinearity or other numerical problems.\
        '%results.condition_number)
    
Teste_VIF()

# Residuos Gráficos
residuos = results.resid
fig, ax = plt.subplots(2,2,figsize=(15,6))
residuos.plot(title="Resíduos do modelo", ax=ax[0][0])
sns.distplot(residuos,ax=ax[0][1])
plot_acf(residuos,lags=40, ax=ax[1][0])
qqplot(residuos,line='s', ax=ax[1][1]);

### ANOVA ###
y1 = df.iloc[:,0]
x1 = df.iloc[:,1]
x2 = df.iloc[:,2]
x3 = df.iloc[:,3]
model = ols('y1 ~ x1 + x2 + x3', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

anova_table(aov_table)

shapiro_test = stats.shapiro(model.resid)
shapiro_test.statistic
shapiro_test.pvalue

# =============================================================================
# REGRESSÃO PARA DADOS DE PAINEL - Parei Aqui
# =============================================================================
y = df.iloc[:,0]
x1 = df.iloc[:,1]
x2 = df.iloc[:,2]
x3 = df.iloc[:,3]
df.isna().sum()
df.describe()

# EFEITOS FIXOS/FIXED EFFECT MODELS
mod = PanelOLS.from_formula('y ~ 1 + x1 + x2 + x3',df)
fe_res = mod.fit(cov_type='clustered', cluster_time=True)
print(fe_res)

# REGRESSÃO COM DADOS EM PAINEL 
# EFEITOS RANDOMICOS/RANDOM EFFECTS
model_re = RandomEffects.from_formula('y ~ 1 + x1 + x2 + x3',df)
model_re = RandomEffects(endog, exog)
re_res = model_re.fit() 
#print results
print(re_res)

# =============================================================================
# TESTE DE HAUSMAN 
# =============================================================================
def hausman(fe, re):

  b = fe.params
  B = re.params
  v_b = fe.cov
  v_B = re.cov
  df = b[np.abs(b) < 1e8].size
  chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B)) 
  
  pval = stats.chi2.sf(chi2, df)
  return chi2, df, pval

hausman_results = hausman(fe_res, re_res) 
print('chi-Squared: ' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value: ' + str(hausman_results[2]))




