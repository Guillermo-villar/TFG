#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:23:54 2023

@author: fran
"""
# basic python libraries
import pandas as pd
import numpy as np
import pdb

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib import style
style.use('fivethirtyeight')
# import matplotlib.animation as animation

import matplotlib.dates as mdates
import matplotlib.colors

font = {'family' : 'Calibri',
        'weight' : 'bold',
        'size'   : 28}

marker_style =[("o", 80), ("o", 200), ('s', 100), ('*', 500), ("v", 500), ('v', 800)]
matplotlib.rc('font', **font)

cm = plt.get_cmap('tab20')
# cm = plt.get_cmap('RdYlGn')
cm_sev =  plt.get_cmap('RdYlBu') #  plt.get_cmap('gist_rainbow') # plt.get_cmap('jet') # 
# cm = plt.get_cmap('gist_rainbow') # plt.get_cmap('tab20')
# cm = plt.get_cmap('jet')

colors = [cm(clr) for clr in range(14)]
colors_sev = [cm_sev(16*clr) for clr in range(6)]

cmap=matplotlib.colors.ListedColormap(colors)
cmap_sev=matplotlib.colors.ListedColormap(colors_sev)

def get_seq_events(df, timestamp_col = None, start_time=None, end_time=None, max_event_sep=5*60):
    
    if timestamp_col == None:
        timestamp_col = 'timestamp'
        
    if start_time == None:
        start_time = df[timestamp_col].iloc[0]
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time, format='%Y-%m-%d')
        
    if end_time == None:
        end_time = df[timestamp_col].iloc[-1]
    elif isinstance(end_time, str):
        end_time = pd.to_datetime(end_time, format='%Y-%m-%d')
    
    DeltaT = pd.Timedelta(seconds=max_event_sep)
    
    id_s = np.where(1-(df[timestamp_col].diff() < DeltaT))[0]
    length_s = np.append(np.diff(id_s),df.shape[0]-id_s[-1])
    
    return id_s, length_s

def df_plot(df, xcolumn=None, ycolumns=None):
    
    df = df.replace(np.nan,0)
    
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1,1,1)
    
    if xcolumn == None:
        xcolumn = 'Time'
        
    if ycolumns == None:
        ycolumns = ['Severity'] # df.columns[2:]
        
    plt.yscale('linear') # 'log')
    
    if str(df[xcolumn].dtype).startswith('datetime'):
        DeltaT=df[xcolumn].iloc[-1]-df[xcolumn].iloc[0]
        if DeltaT.days > 100:
            xdata = df.index
        else:   
            xdata = mdates.date2num(df[xcolumn])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
    lines = []
    label_legend = []
    for column in ycolumns:
        label_legend.append(column)
        lines.append(ax.plot(xdata, df[column], linewidth=2, label=column))
        # lines.append(ax.plot(xdata, np.log10(df[column].astype(float)+1), linewidth=2, label=column))

    legend_elements = [i[0] for i in lines[::1]]

    ax.legend(handles=legend_elements)
    plt.gcf().autofmt_xdate()
    
def df_plot_events(df, timestamp_col = None,
                   element_levels_dict = None, alarm_level_dict=None):
    
    if timestamp_col == None:
        timestamp_col = 'timestamp'
        
    if element_levels_dict == None:
        elements_levels = [('Incidence', 18), ('Site/Energy/Climate', 16), 
                           ('Service Alarm', 14), ('Multiple Services', 12), 
                           ('DTT Monitor', 10), 
                           ('UCA RF Switch', 8), ('Chassis/GPS', 6), ('Transmitter', 4), 
                           ('SSW Data Switch', 2), ('IPA Data Switch', 0)]
        element_levels_dict = dict([(k, v) for k, v in elements_levels])
        
    if alarm_level_dict == None:
      alarm_levels = [('Critical', 24), # ('Critical Low', 22), ('Major High', 20), 
                      ('Major', 18), # ('Major Low', 16), 
                      ('Minor', 14), # ('Minor Low', 12), 
                      # ('Warning High', 10), 
                      ('Warning', 8), # ('Warning Low', 6), 
                      ('Timeout', 4), # ('Notice', 2), 
                      ('Normal', 1)]
      alarm_level_dict = dict([(k, v) for k, v in alarm_levels])
        
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1,1,1)
    plt.yscale('linear') # 'log')
    xcolumn = timestamp_col
    if str(df[xcolumn].dtype).startswith('datetime'):
        DeltaT=df[xcolumn].iloc[-1]-df[xcolumn].iloc[0]
        if DeltaT.days > 100:
            xdata = df.index
        else:   
            xdata = mdates.date2num(df[xcolumn])
            # xdata_attack = mdates.date2num(df_attack[xcolumn])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
    color_map=plt.get_cmap('RdYlGn')

    for key, value in alarm_level_dict.items():
      idx_sel = np.where(df['Severity']==key)[0]
      if len(idx_sel) > 0:
        color = color_map(3/value)
        # markers,stems,base = ax.stem(xdata[idx_sel], df.iloc[idx_sel]['Severity'].map(level_dict).astype(float), label=key)
        # plt.setp(stems, 'linewidth', 8)
        ax.scatter(xdata[idx_sel], df.iloc[idx_sel]['Severity'].map(alarm_level_dict).astype(float), label=key, color=color)
    ax.legend()
    
def df_plot_seq_events(alarms_df, incs_df=None, alarms_incs_out=None, 
                       alarm_timestamp_col = None, inc_timestamp_col = None,
                       Sites = None,
                       Services=None, # Service Name list: mpe1, mpe2
                       Elements=None, # SIRA ID list
                       min_seq_event=5, max_event_sep=5*60, 
                       start_time=None, end_time=None,
                       services_dict = None,
                       elements_dict = None,
                       sites_topology = None,
                       element_levels_dict = None,
                       alarm_level_dict = None,
                       verbose = False):
    
    
    df = alarms_df
    
    if alarm_timestamp_col == None:
        alarm_timestamp_col = 'TimeOfArrival'

    if inc_timestamp_col == None:
        inc_timestamp_col = 'timestamp'
        
    if start_time == None:
        start_time = df[alarm_timestamp_col].iloc[0]
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time, format='%Y-%m-%d')
        
    if end_time == None:
        end_time = df[alarm_timestamp_col].iloc[-1]
    elif isinstance(end_time, str):
        end_time = pd.to_datetime(end_time, format='%Y-%m-%d')


    df_sel_timestamp = df[(df[alarm_timestamp_col] >= start_time) & (df[alarm_timestamp_col] <= end_time)]
    
    if incs_df is not None:
        incs_sel_timestamp = incs_df[(incs_df[inc_timestamp_col] >= start_time) & (incs_df[inc_timestamp_col] <= end_time)]
        
    if alarms_incs_out is not None:
        alarms_incs_out_timestamp = alarms_incs_out[(alarms_incs_out['timestamp_i'] >= start_time) & (alarms_incs_out['timestamp_i'] <= end_time)]
    
    if Elements == None:
        Elements = [[] for site in Sites]
        nsite = 0
        for site in Sites:
            nserv = 0
            if verbose:
                print()
                print(site)
            for service in Services:
                if service != 'energy':
                    if verbose:
                        print(service)
                    Elements[nsite].append(sites_topology[site][service]['Elements'])
                    if verbose:
                        for elem_dict in elements_dict:
                            for elem in Elements[nsite][nserv]:
                                if elem == elem_dict: # elem == str(elem_dict['ResourceSIRAId']):
                                    print(elem_dict, ': ', elements_dict[elem_dict]['Name'])
                nserv += 1
                if verbose:
                    print()
            nsite += 1
            
        
    if element_levels_dict == None:
        elements_levels = [('Incidence', 18), ('Site/Energy/Climate', 16), 
                           ('Service Alarm', 14), ('Multiple Services', 12), 
                           ('DTT Monitor', 10), 
                           ('UCA RF Switch', 8), ('Chassis/GPS', 6), ('Transmitter', 4), 
                           ('SSW Data Switch', 2), ('IPA Data Switch', 0)]
        element_levels_dict = dict([(k, v) for k, v in elements_levels])
    
    if alarm_level_dict == None:
        alarm_levels = [('Critical', 24), # ('Critical Low', 22), ('Major High', 20), 
                        ('Major', 18), # ('Major Low', 16), 
                        ('Minor', 14), # ('Minor Low', 12), 
                        # ('Warning High', 10), 
                        ('Warning', 8), # ('Warning Low', 6), 
                        ('Timeout', 4), # ('Notice', 2), 
                        ('Normal', 1)]
        alarm_level_dict = dict([(k, v) for k, v in alarm_levels])
        
        
    Serv_ext = [serv for serv in Services]
    Serv_ext.append('Multiple Services')
    Serv_ext.append('energy')
    Serv_ext.append('unknown')
    Serv_ext.append('Service Alarm')
    Serv_ext.append('Site/Energy/Climate')
    Serv_ext.append('Incidence')
    
    Serv_cmap = {Serv_ext[ns]: colors[ns] for ns in range(len(Serv_ext))}


    if len(Services) == 1:
        # Define a dictionary mapping severity levels to colors
        severity_color_dict = {
            'critical': 'red',
            'major': 'orange',
            'minor': 'yellow',
            'warning': 'blue',
            'timeout': 'purple',
            'normal': 'green'
        }
        
        # Map severity labels to colors using the dictionary
        df_sel_timestamp['SeverityColor'] = df_sel_timestamp['AlarmState'].str.lower().map(severity_color_dict)

    Plot = [[] for site in Sites]
    nsite = 0
    for site in Sites:
        Plot[nsite] = [[] for service in Serv_ext]
        nsite += 1
        
    nsite = 0
    for site in Sites:
        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='x', labelrotation = 45)
        plt.yscale('linear') # 'log')
        xcolumn = alarm_timestamp_col
                            
        ax.set_yticks(list(element_levels_dict.values()))
        ax.set_yticklabels(list(element_levels_dict.keys()))
   
        label_legend = [] 
        df_sel_site = df_sel_timestamp[df_sel_timestamp['SiteCodInfo'].astype(str)==site]
        if incs_df != None:
            if incs_df.shape[0] > 0: 
                incs_sel_site = incs_sel_timestamp[incs_sel_timestamp['I. Centro Afectado'].astype(str)==site]
        if alarms_incs_out != None:
            if alarms_incs_out.shape[0] > 0: 
                alarms_incs_sel_site = alarms_incs_out_timestamp[alarms_incs_out_timestamp['site_a'].astype(str)==site]
        
        nserv = 0
        label_added = [False] * len(Services)
        label_added_multserv = False
        label_added_all = False
        
        for service in Services:
            # pdb.set_trace()
            key_single_service=df_sel_site['Services']==service
            key_multiple_service=df_sel_site['Services'].str.contains(service)
            key_sel_serv = key_multiple_service
            
            if any(key_sel_serv):
                df_sel = df_sel_site[key_sel_serv]
                mask_service = df_sel['ElemServ']==0
                if any(mask_service):
                    df_sel_serv = df_sel[mask_service]
                    xdata_serv = mdates.date2num(df_sel_serv[xcolumn])
                    if verbose:
                        for serv in services_dict:
                            mask = df_sel_serv['ElemServID']==serv['ServiceId']
                            
                            if any(mask) and verbose:
                                print()
                                print(serv)
                                print()
                    df_type_serv =  df_sel_serv['ElemServID'].apply(lambda x: 'Service Alarm')
                    
                    if label_added[nserv]:
                        Plot[nsite][nserv] = ax.scatter(xdata_serv, df_type_serv.map(element_levels_dict).astype(float) , 
                                                        c=df_sel_serv['Services'].map(Serv_cmap), 
                                                        marker=marker_style[1][0],
                                                        s=marker_style[1][1])
                    else:
                        Plot[nsite][nserv] = ax.scatter(xdata_serv, df_type_serv.map(element_levels_dict).astype(float) , 
                                                        c=df_sel_serv['Services'].map(Serv_cmap), 
                                                        marker=marker_style[1][0],
                                                        s=marker_style[1][1],
                                                        label=service)    
                        label_added[nserv] = True
                
                mask_elem = df_sel['ElemServ']==1
                df_sel_elem = df_sel[mask_elem]
                for element in Elements[nsite][nserv]:
                    key=df_sel_elem['ElemServID'].astype(str)==element
                    if any(key):
                        xdata_elem = mdates.date2num(df_sel_elem[key][xcolumn])
                        df_type_elem = df_sel_elem[key]['ElemServID'].apply(lambda x: elements_dict[str(x)]['Type'])
                        df_type_serv = df_sel_elem[key]['ElemServID'].apply(lambda x: service)
                        if label_added[nserv]:
                            if len(Services) == 1:
                                Plot[nsite][nserv] = ax.scatter(xdata_elem, df_type_elem.map(element_levels_dict).astype(float), 
                                                                # # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                # c = df_type_serv.map(Serv_cmap),
                                                                c=df_sel_elem[key]['SeverityColor'], 
                                                                marker=marker_style[0][0],
                                                                s=marker_style[0][1])
                            else:
                                Plot[nsite][nserv] = ax.scatter(xdata_elem, df_type_elem.map(element_levels_dict).astype(float), 
                                                                # # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                c = df_type_serv.map(Serv_cmap),
                                                                # c=df_sel_elem[key]['SeverityColor'], 
                                                                marker=marker_style[0][0],
                                                                s=marker_style[0][1])
                                
                        else:
                            if len(Services) == 1:
                                Plot[nsite][nserv] = ax.scatter(xdata_elem, df_type_elem.map(element_levels_dict).astype(float), 
                                                                # # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                # c = df_type_serv.map(Serv_cmap),
                                                                c=df_sel_elem[key]['SeverityColor'], 
                                                                marker=marker_style[0][0],
                                                                s=marker_style[0][1],
                                                                label=service)
                            else:
                                Plot[nsite][nserv] = ax.scatter(xdata_elem, df_type_elem.map(element_levels_dict).astype(float), 
                                                                # # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                c = df_type_serv.map(Serv_cmap),
                                                                # c=df_sel_elem[key]['SeverityColor'], 
                                                                marker=marker_style[0][0],
                                                                s=marker_style[0][1],
                                                                label=service)
                            label_added[nserv] = True

                mask_multiserv = df_sel_elem['Services'].str.split(', ').apply(lambda x: True if len(x) > 1 else False)
                
                if any(mask_multiserv):
                    df_sel_multiserv = df_sel_elem[mask_multiserv]
        
                    for element in Elements[nsite][nserv]:
                        key=df_sel_multiserv['ElemServID'].astype(str)==element
                        if any(key):
                            xdata_multiserv = mdates.date2num(df_sel_multiserv[key][xcolumn])
                            df_type_multiserv_elem = df_sel_multiserv[key]['ElemServID'].apply(lambda x: elements_dict[str(x)]['Type'])
                            df_type_multiserv_serv = df_sel_multiserv[key]['ElemServID'].apply(lambda x: 'Multiple Services')
                            if label_added_multserv:
                                Plot[nsite][nserv] = ax.scatter(xdata_multiserv, df_type_multiserv_elem.map(element_levels_dict).astype(float), 
                                                                # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                c = df_type_multiserv_serv.map(Serv_cmap),
                                                                marker=marker_style[2][0],
                                                                s=marker_style[2][1])
                            else:
                                Plot[nsite][nserv] = ax.scatter(xdata_multiserv, df_type_multiserv_elem.map(element_levels_dict).astype(float), 
                                                                # c=df_sel_elem[key]['Services'].map(Serv_cmap),
                                                                c = df_type_multiserv_serv.map(Serv_cmap),
                                                                marker=marker_style[2][0],
                                                                s=marker_style[2][1],
                                                                label='Multiple Services')
                                label_added_multserv = True

            nserv += 1
        
        mask_ukn = df_sel_site['Services'].str.contains('unknown').fillna(False)
        if any(mask_ukn):
            df_sel_ukn = df_sel_site[mask_ukn]
            xdata_ukn = mdates.date2num(df_sel_ukn[xcolumn])
            df_type_ukn = df_sel_ukn['ElemServID'].apply(lambda x: 'Service')
            ax.scatter(xdata_ukn, df_type_ukn.map(element_levels_dict).astype(float), 
                        c=df_sel_ukn['Services'].map(Serv_cmap),
                        marker=marker_style[1][0],
                        s=marker_style[1][1], label='Unknown')
                                                          

        mask_energy = df_sel_site['Services'].str.contains('energy').fillna(False)  
        if any(mask_energy):
            # mask_site = mask_all ^ mask_energy        
            df_sel_energy = df_sel_site[mask_energy]
            xdata_energy = mdates.date2num(df_sel_energy[xcolumn])
            df_energy_type =  df_sel_energy['ElemServID'].apply(lambda x: 'Site/Energy/Climate')
            # label_legend.append('Energy')
            ax.scatter(xdata_energy, df_energy_type.map(element_levels_dict).astype(float), 
                       c=df_energy_type.map(Serv_cmap), 
                       marker=marker_style[1][0],
                       s=marker_style[1][1],
                       label='Site/Energy/Climate')
        
        # mask_energy = (df_sel_timestamp['serv_serv_ci']==0) * (df_sel_timestamp['SIRA ID']==0) * (df_sel_timestamp['Parameter key'].str.contains('energy'))
        All_Serv_O_I ='mpe1, mpe2, mpe3, mpe4, mpe5, rge2, pilo4k'
        
        # pdb.set_trace()
        # mask_all = df_sel_site['Services'] == All_Serv_O_I
        # mask_all = df_sel_site['Services'].str.contains(All_Serv_O_I)
        mask_all = df_sel_site['Services'].apply(lambda x: set(x.split(', ')).issuperset(set(All_Serv_O_I.split(', '))))

        if any(mask_all):
            df_sel_all = df_sel_site[mask_all]
            key_all_ElemServ = df_sel_all['Services'].apply(lambda x: False)
            for element in elements_dict.keys(): # Elements[nsite][nserv]:
                key=df_sel_all['ElemServID'].astype(str)==element
                if any(key):
                    key_all_ElemServ += key
                    xdata_all = mdates.date2num(df_sel_all[key][xcolumn])
                    df_type_all = df_sel_all[key]['ElemServID'].apply(lambda x: elements_dict[str(x)]['Type'])
                    df_serv_all = df_sel_all[key]['ElemServID'].apply(lambda x: 'Multiple Services')
                    if label_added_all:
                        Plot[nsite][nserv] = ax.scatter(xdata_all, df_type_all.map(element_levels_dict).astype(float), 
                                                        c=df_serv_all.map(Serv_cmap),
                                                        marker=marker_style[3][0],
                                                        s=marker_style[3][1])
                    else:
                        Plot[nsite][nserv] = ax.scatter(xdata_all, df_type_all.map(element_levels_dict).astype(float), 
                                                        c=df_serv_all.map(Serv_cmap),
                                                        marker=marker_style[3][0],
                                                        s=marker_style[3][1],
                                                        label='All Services')
                        label_added_all = True
                
            """    
            # Select alarms affecting all services (excluding energy)
            key = ~df_sel_all[~key_all_ElemServ]['ParameterKey'].str.contains('energy').fillna(False)
            if any(key):
                df_sel_all_other = df_sel_all[~key_all_ElemServ][key]
                xdata_all_other = mdates.date2num(df_sel_all_other[xcolumn])
                df_type_all_other = df_sel_all_other['ElemServID'].apply(lambda x: 'Multiple Services')
                Plot[nsite][nserv] = ax.scatter(xdata_all_other, df_type_all_other.map(element_levels_dict).astype(float), 
                                                c=df_sel_all_other['Services'].map(Serv_cmap),
                                                marker=marker_style[3][0],
                                                s=marker_style[3][1], label='All Services')
            """
        if incs_df != None:
            if incs_df.shape[0] > 0: #  is not None:
                x_column_inc = inc_timestamp_col
                xdata_incs = mdates.date2num(incs_sel_site[x_column_inc])
                incs_type =  incs_sel_site[x_column_inc].apply(lambda x: 'Incidence')
                Plot[nsite][nserv] = ax.scatter(xdata_incs, incs_type.map(element_levels_dict).astype(float), 
                                                   c=incs_type.map(Serv_cmap),
                                                   marker=marker_style[1][0],
                                                   s=marker_style[1][1],
                                                   label='Incidence')
            
        if alarms_incs_out != None:
            if alarms_incs_out.shape[0] > 0: #  is not None:
                # Tagged Alarms and Incidences
                tagged_alarms_df=alarms_incs_sel_site[['timestamp_a', 'id_a', 'elem_name_a', 'serv_name_a', 'type_a', 'else_id', 'site_a', 'severity_a']]
                xcolumn_a = 'timestamp_a'
                tagged_elements_df = tagged_alarms_df[tagged_alarms_df['type_a']==1]
                xdata_tag_elements = mdates.date2num(tagged_elements_df[xcolumn_a])
                # df_type = tagged_elements_df['else_id'].apply(lambda x: elements_dict[str(x).zfill(8)]['Type'])
                if  tagged_elements_df['else_id'].str.startswith('000').any():
                    # pdb.set_trace()
                    # df_type = tagged_elements_df['else_id'].str.removeprefix('000').apply(lambda x: elements_dict[x]['Type'])
                    df_type = tagged_elements_df['else_id'].str.lstrip('0').apply(lambda x: elements_dict[x]['Type'])
                # elif tagged_elements_df['else_id'].str.startswith('0').any():
                #     df_type = tagged_elements_df['else_id'].str.removeprefix('0').apply(lambda x: elements_dict[x]['Type'])
                else:
                    df_type = tagged_elements_df['else_id'].apply(lambda x: elements_dict[str(x)]['Type'])
                      
                tagged_services_df = tagged_alarms_df[tagged_alarms_df['type_a']==2]
                xdata_tag_services = mdates.date2num(tagged_services_df[xcolumn_a])
                df_serv_type =  tagged_services_df['else_id'].apply(lambda x: 'Service Alarm') 
                tagged_incs_df = alarms_incs_sel_site[['timestamp_i', 'id_i', 'serv_name_i', 'site_i','urgency_i', 'impact_i', 'priority_i']]
                xcolumn_i = 'timestamp_i'
                xdata_tag_incs = mdates.date2num(tagged_incs_df[xcolumn_i])
                df_incs_type =  tagged_incs_df['serv_name_i'].apply(lambda x: 'Incidence') 
    
                
                service = 'Multiple Services'
                mask_all_elem = tagged_elements_df['serv_name_a'].apply(lambda x: set(x.split(', ')).issuperset(set(All_Serv_O_I.split(', '))))
                ax.scatter(xdata_tag_elements[mask_all_elem], df_type[mask_all_elem].map(element_levels_dict).astype(float), 
                           color=Serv_cmap[service],
                           marker=marker_style[4][0],
                           s=marker_style[4][1],
                           zorder=0)   
                mask_all_alarm = tagged_alarms_df['serv_name_a'].apply(lambda x: set(x.split(', ')).issuperset(set(All_Serv_O_I.split(', '))))
                ax.scatter(xdata_tag_incs[mask_all_alarm], df_incs_type[mask_all_alarm].map(element_levels_dict).astype(float) , 
                           color=Serv_cmap[service],
                           marker=marker_style[5][0],
                           s=marker_style[5][1],
                           zorder=0)
                
                for service in Services:
                    mask = tagged_elements_df['serv_name_a'] == service # .str.contains(service) # 
                    ax.scatter(xdata_tag_elements[mask], df_type[mask].map(element_levels_dict).astype(float), 
                               color=Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               zorder=0)
                    
                    mask = tagged_services_df['serv_name_a'] == service #.str.contains(service) # ==service
                    ax.scatter(xdata_tag_services[mask], df_serv_type[mask].map(element_levels_dict).astype(float), 
                               color=Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               zorder=0)
                    
                    mask = tagged_alarms_df['serv_name_a'] == service #.str.contains(service) # ==service
                    ax.scatter(xdata_tag_incs[mask], df_incs_type[mask].map(element_levels_dict).astype(float) , 
                               color=Serv_cmap[service],
                               marker=marker_style[5][0],
                               s=marker_style[5][1],
                               zorder=0)
                
                # pdb.set_trace()
                service = 'energy'
                mask_energy_elem = tagged_elements_df['serv_name_a'] == service
                ax.scatter(xdata_tag_elements[mask_energy_elem], df_type[mask_energy_elem].map(element_levels_dict).astype(float), 
                           color=Serv_cmap[service],
                           marker=marker_style[4][0],
                           s=marker_style[4][1],
                           zorder=0)   
                mask_energy_serv = tagged_alarms_df['serv_name_a'] == service
                ax.scatter(xdata_tag_incs[mask_energy_serv], df_incs_type[mask_energy_serv].map(element_levels_dict).astype(float) , 
                           color=Serv_cmap[service],
                           marker=marker_style[5][0],
                           s=marker_style[5][1],
                           zorder=0)
            
        else:
            # Tagged Alarms
            tagged_alarms_df = df_sel_site[df_sel_site['target'] == 1]
            xcolumn_a = 'TimeOfArrival'
            tagged_elements_df = tagged_alarms_df[tagged_alarms_df['ElemServ']==1]
            xdata_tag_elements = mdates.date2num(tagged_elements_df[xcolumn_a])
            
            tagged_services_df = tagged_alarms_df[tagged_alarms_df['ElemServ']==0]
            xdata_tag_services = mdates.date2num(tagged_services_df[xcolumn_a])
            
            if  tagged_elements_df['ElemServID'].str.startswith('000').any():
                # pdb.set_trace()
                # df_type = tagged_elements_df['else_id'].str.removeprefix('000').apply(lambda x: elements_dict[x]['Type'])
                df_type = tagged_elements_df['ElemServID'].str.lstrip('0').apply(lambda x: elements_dict[x]['Type'])
            # elif tagged_elements_df['else_id'].str.startswith('0').any():
            #     df_type = tagged_elements_df['else_id'].str.removeprefix('0').apply(lambda x: elements_dict[x]['Type'])
            else:
                df_type = tagged_elements_df['ElemServID'].apply(lambda x: elements_dict[str(x)]['Type'])
            
            tagged_services_df = tagged_alarms_df[tagged_alarms_df['ElemServ']==0]
            xdata_tag_services = mdates.date2num(tagged_services_df[xcolumn_a])
            df_serv_type =  tagged_services_df['ElemServID'].apply(lambda x: 'Service Alarm') 
            
            for service in Services:
                mask = tagged_elements_df['Services'] == service # .str.contains(service) # 
                if len(Services) == 1:
                    ax.scatter(xdata_tag_elements[mask], df_type[mask].map(element_levels_dict).astype(float), 
                               c=tagged_elements_df.loc[mask]['SeverityColor'], # Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               # label=tagged_elements_df.loc[mask]['AlarmState'],
                               zorder=0)
                    mask = tagged_services_df['Services'] == service #.str.contains(service) # ==service
                    ax.scatter(xdata_tag_services[mask], df_serv_type[mask].map(element_levels_dict).astype(float), 
                               c=tagged_services_df.loc[mask]['SeverityColor'], # Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               zorder=0)
                else:
                    ax.scatter(xdata_tag_elements[mask], df_type[mask].map(element_levels_dict).astype(float), 
                               color=Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               # label=tagged_elements_df.loc[mask]['AlarmState'],
                               zorder=0)
                
                    mask = tagged_services_df['Services'] == service #.str.contains(service) # ==service
                    ax.scatter(xdata_tag_services[mask], df_serv_type[mask].map(element_levels_dict).astype(float), 
                               color=Serv_cmap[service],
                               marker=marker_style[4][0],
                               s=marker_style[4][1],
                               zorder=0)
                
            service = 'energy'
            mask_energy_elem = tagged_elements_df['Services'] == service
            
            ax.scatter(xdata_tag_elements[mask_energy_elem], df_type[mask_energy_elem].map(element_levels_dict).astype(float), 
                       color=Serv_cmap[service], #tagged_elements_df.loc[mask_energy_elem]['SeverityColor'], # 
                       marker=marker_style[4][0],
                       s=marker_style[4][1],
                       zorder=0)   

            
        # if incs_df.shape[0] > 0 or alarms_incs_out.shape[0] > 0:
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        if incs_df != None and alarms_incs_out != None:
            if incs_df.shape[0] > 0 or alarms_incs_out.shape[0] > 0:
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        
        if len(Services) == 1:
            # Create proxy artists for the legend
            legend_handles = [Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=label) 
                              for label, color in severity_color_dict.items()]
            
            # Combine the legend handles
            all_handles = handles + legend_handles
        
            all_labels = labels + list(severity_color_dict.keys())

            # Add legend for severity colors
            ax.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.6),
                      ncol=4, fancybox=True, shadow=True, title='Severity')

    
        else:
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6),
                      ncol=4, fancybox=True, shadow=True)
            # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6),
            #           ncol=4, fancybox=True, shadow=True)
        ax.set_title(label= 'Site: '+site+'. Services: '+', '.join(Services))
            
        DeltaT=end_time-start_time

        if DeltaT.days > 20:
            xfmt=mdates.DateFormatter('%d-%m')
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        else:
            xfmt=mdates.DateFormatter('%d-%m %H:%M')
            # set ticks every 6 hours mins 
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=12*60))
            
        ax.xaxis.set_major_formatter(xfmt)
        plt.show()
        nsite += 1
        
        """

            

                # label_legend.append('Tagged Incidence')
        """
        """
        ax.set_yticks(list(element_levels_dict.values()))
        ax.set_yticklabels(list(element_levels_dict.keys()))
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6),
                  ncol=4, fancybox=True, shadow=True)
        # ax.legend(label_legend, loc='lower center', bbox_to_anchor=(0.5, -0.6),
        #           ncol=4, fancybox=True, shadow=True)
        ax.set_title(label= 'Site: '+site+'. Services: '+', '.join(Services))
        """
    plt.rcdefaults()