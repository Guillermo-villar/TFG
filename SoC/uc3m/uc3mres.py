#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:12 2023

@author: SOCTRA Team
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries 
import pandas as pd
import logging

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_print_results(df_res, type_res='plain', col_label = 'Label', 
                       col_res='Result', out_path=None,
                       flag_latex=True):
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    df_print = pd.DataFrame()#columns=['Project','Patterns','IR','Acc', 'BAcc', 'pFA', 'pM', 'pD'])
    
    if type_res == 'plain':

        project_counts = df_res['proyecto'].value_counts()
        
        res_total = df_res[col_res].value_counts()
        ref_total = df_res[col_label].value_counts()
        
        if flag_latex:
            print('\\begin{center}')
            print('\\begin{tabular}{r|ccccc}')
            print('\\hline ')
            print('Project (Patterns$|$IR) & $Acc$ & $BAcc$ & $p_{FA}$ & $p_{M}$ & $p_D$ \\\\ \\hline')
        
        for project in project_counts.index:
            res_project = df_res[df_res['proyecto']==project][col_res].value_counts()
            ref_project = df_res[df_res['proyecto']==project][col_label].value_counts()
            if 'False Alarm' in res_project.index:
                pFA = res_project['False Alarm'] / ref_project[False]
            else:
                pFA = 0
                
            if ('No Detection' in res_project.index) & (True in ref_project.index):    
                pM = res_project['No Detection'] / ref_project[True]
            else:
                pM = 0
                    
            if 'Accurate' in res_project.index:
                Acc = res_project['Accurate'] / project_counts[project]
            else:
                Acc = 0
                
            BAcc = 1 - (pFA+pM)/2
            if True not in ref_project.index:
                str_project_ocu = '%s (%d$|$Inf)'%(project,project_counts[project])
            elif False not in ref_project.index:
                str_project_ocu = '%s (%d$|$0)'%(project,project_counts[project])    
            else:
                str_project_ocu = '%s (%d$|$%2.2f)'%(project,project_counts[project],ref_project[False]/ref_project[True])
            
            if flag_latex:
                print('%27s & %2.5f & %2.5f & %2.5f & %2.5f & %2.5f \\\\'%(str_project_ocu,Acc,BAcc,pFA,pM,1-pM))
            
            if True not in ref_project.index:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': 'Inf',
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
            elif False not in ref_project.index:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': 0,
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
            else:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': ref_project[False]/ref_project[True],
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
                
                    
                       
            if df_print.shape[0]==0:
                df_print=pd.DataFrame([dict_print])
            else:
                df_row=pd.DataFrame([dict_print])
                df_print = pd.concat([df_print,df_row], ignore_index=True)
            
            
        if 'False Alarm' in res_total.index:
            pFA = res_total['False Alarm'] / ref_total[False]
        else:
            pFA = 0
            
        if 'No Detection' in res_total.index:    
            pM = res_total['No Detection'] / ref_total[True]
        else:
            pM = 0
                
        if 'Accurate' in res_total.index:
            Acc = res_total['Accurate'] / df_res.shape[0]
        else:
            Acc = 0
            
        BAcc = 1 - (pFA+pM)/2
        str_poject_ocu = '%s (%d$|$%2.2f)'%('Total',df_res.shape[0],ref_total[False]/ref_total[True])
        
        if flag_latex:
            print('\\hline')
            print('%27s & %2.5f & %2.5f & %2.5f & %2.5f & %2.5f \\\\'%(str_poject_ocu,Acc,BAcc,pFA,pM,1-pM))    
            print('\\hline ')
            print('\\end{tabular}')
            print('\\end{center}')
        
        dict_print = {'Project': 'Total', 
                      'Patterns': df_res.shape[0],
                      'IR': ref_total[False]/ref_total[True],
                      'Acc': Acc,
                      'BAcc': BAcc,
                      'pFA': pFA,
                      'pM': pM,
                      'pD': 1-pM}
        
        df_row=pd.DataFrame([dict_print])
        df_print = pd.concat([df_print,df_row], ignore_index=True)
        
        if out_path != None:
            df_print.to_excel(out_path,index=False)
            logger.info('Results saved to {}'.format(out_path))
        
        
    elif type_res == 'weighted':

        project_counts = df_res['proyecto'].value_counts()
        res_total = df_res[col_res].value_counts()
        ref_total = df_res[col_label].value_counts()
        
        if flag_latex:                           
            print('\\begin{center}')
            print('\\begin{tabular}{r|ccccc}')
            print('\\hline ')
            print('Project (Patterns$|$IR) & $Acc$ & $BAcc$ & $p_{FA}$ & $p_{M}$ & $p_D$ \\\\ \\hline')
        
        for project in project_counts.index:
            res_project = df_res[df_res['proyecto']==project][col_res].value_counts()
            ref_project = df_res[df_res['proyecto']==project][col_label].value_counts()               
            df_project = df_res[df_res['proyecto']==project]
            if 'False Alarm' in res_project.index:                    
                pFA = df_project[df_project[col_res]=='False Alarm']['weight'].sum()/df_project[df_project[col_label]==False]['weight'].sum()
            else:
                pFA = 0
                
            if 'No Detection' in res_project.index:    
                pM = df_project[df_project[col_res]=='No Detection']['weight'].sum()/df_project[df_project[col_label]==True]['weight'].sum()
            else:
                pM = 0
                    
            if 'Accurate' in res_project.index:                    
                Acc = df_project[df_project[col_res]=='Accurate']['weight'].sum()/df_project['weight'].sum()
            else:
                Acc = 0
                
            BAcc = 1 - (pFA+pM)/2
            if True not in ref_project.index:
                str_project_ocu = '%s (%d$|$Inf)'%(project,project_counts[project])
            elif False not in ref_project.index:
                str_project_ocu = '%s (%d$|$0)'%(project,project_counts[project])    
            else:
                str_project_ocu = '%s (%d$|$%2.2f)'%(project,project_counts[project],ref_project[False]/ref_project[True])
            
            
            if flag_latex:
                print('%27s & %2.5f & %2.5f & %2.5f & %2.5f & %2.5f \\\\'%(str_project_ocu,Acc,BAcc,pFA,pM,1-pM))
            
            
            if True not in ref_project.index:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': 'Inf',
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
            elif False not in ref_project.index:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': 0,
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
            else:
                dict_print = {'Project': project, 
                              'Patterns': project_counts[project],
                              'IR': ref_project[False]/ref_project[True],
                              'Acc': Acc,
                              'BAcc': BAcc,
                              'pFA': pFA,
                              'pM': pM,
                              'pD': 1-pM}
                
                    
                       
            if df_print.shape[0]==0:
                df_print=pd.DataFrame([dict_print])
            else:
                df_row=pd.DataFrame([dict_print])
                df_print = pd.concat([df_print,df_row], ignore_index=True)
            
            
        if 'False Alarm' in res_total.index:
            pFA = df_res[df_res[col_res]=='False Alarm']['weight'].sum()/df_res[df_res[col_label]==False]['weight'].sum()
        else:
            pFA = 0
            
        if 'No Detection' in res_total.index:    
            pM = df_res[df_res[col_res]=='No Detection']['weight'].sum()/df_res[df_res[col_label]==True]['weight'].sum()
        else:
            pM = 0
                
        if 'Accurate' in res_total.index:
            Acc = df_res[df_res[col_res]=='Accurate']['weight'].sum()/df_res['weight'].sum()
        else:
            Acc = 0
            
        BAcc = 1 - (pFA+pM)/2
        str_poject_ocu = '%s (%d$|$%2.2f)'%('Total',df_res.shape[0],ref_total[False]/ref_total[True])
        
        if flag_latex:
            print('\\hline')
            print('%27s & %2.5f & %2.5f & %2.5f & %2.5f & %2.5f \\\\'%(str_poject_ocu,Acc,BAcc,pFA,pM,1-pM))    
            print('\\hline ')
            print('\\end{tabular}')
            print('\\end{center}')
        
        
        dict_print = {'Project': 'Total', 
                      'Patterns': df_res.shape[0],
                      'IR': ref_total[False]/ref_total[True],
                      'Acc': Acc,
                      'BAcc': BAcc,
                      'pFA': pFA,
                      'pM': pM,
                      'pD': 1-pM}
        
        df_row=pd.DataFrame([dict_print])
        df_print = pd.concat([df_print,df_row], ignore_index=True)
        
        if out_path != None:
            df_print.to_excel(out_path,index=False)
            logger.info('Results saved to {}'.format(out_path))
        