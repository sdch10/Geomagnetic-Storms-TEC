#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:02:28 2019
### STORM STUDY FOR SMALLER SPATIO-TEMPORAL AVERAGING ####
@author: sdch10
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm
import h5py as hf
import datetime as dt
import glob
import pandas as pd
from madrigalWeb import madrigalWeb as md
import matplotlib.colors as colors
import matplotlib
from scipy.stats import spearmanr as spear
from scipy.stats import pearsonr as pearson
import pyIGRF as igrf
from dask import delayed as delayed
import dask.dataframe as dskdf
import time as timeclock
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats, linalg
from astral import Astral 

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr

class MidpointNormalize(colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def makeTECgrid(latitudearr,longitudearr,TECarray,UTstringarr,timechoice):
    ### gives timesnapshot of: 1) meshgrid of longitude 2) meshgrid of latitude, 3)2-D mesh of TEC
    ###  inputs: 1-D latitude array, 1-D longitude array, 1-D timearray (sequential) and the time snapshot desired
    
    latsteps=np.unique(latitudearr)
    lonsteps=np.unique(longitudearr)
    
    lonx,laty=np.meshgrid(lonsteps,latsteps)
    TECgrid=np.zeros([np.size(latsteps),np.size(lonsteps)])
    
    length=np.size(latitudearr)
    for i in range(length):
        if UTstringarr[i]==timechoice:
            longind=np.where(lonsteps==longitudearr[i])[0]
            latind=np.where(latsteps==latitudearr[i])[0]
            TECgrid[latind[0],longind[0]]=TECarray[i]
        else:
            continue
            
    return lonx,laty,TECgrid

    
def getfilename(date):
    yy=date.year
    mm=date.month
    
    if len(str(mm))<2:
        filestr=str(yy)+str(0)+str(mm)
    else:
        filestr=str(yy)+str(mm)
    
    hdf5_base="/home/sdch10/DATA/geomag/symh/hdf5/%d*.h5"
    fname=glob.glob(hdf5_base%int(filestr))[0]
    return fname

def new_check_symh(yy,mm,dd):
    thisdate=dt.datetime(yy,mm,dd,0,0,0)
    day_start_inspect=thisdate+dt.timedelta(days=-3)
    day_end_inspect=thisdate+dt.timedelta(days=4)
    
    ##### READ SYM-H FILES ##############
    
    hdf5_base="/home/sdch10/DATA/geomag/symh/hdf5/%d*.h5"
    
    start_month=day_start_inspect.month
    end_month=day_end_inspect.month
    storm_month=mm
    
    start_yymonth_str=str(yy)+'{:02d}'.format(start_month)
    end_yymonth_str=str(yy)+'{:02d}'.format(end_month)
    storm_yymonth_str=str(yy)+'{:02d}'.format(storm_month)
    
    if start_month==storm_month and not storm_month==end_month:
        label=1
    elif storm_month==end_month and not start_month==storm_month:
        label=2
    else:
        label=3
    
    print(label)
    if label==1:
        fname1=glob.glob(hdf5_base%int(start_yymonth_str))[0]
        o1=pd.read_hdf(fname1, mode="r", key="df")
        fname2=glob.glob(hdf5_base%int(end_yymonth_str))[0]
        o2=pd.read_hdf(fname2, mode="r", key="df")
        o=pd.concat([o1,o2])        
    elif label==2:
        fname1=glob.glob(hdf5_base%int(start_yymonth_str))[0]
        o1=pd.read_hdf(fname1, mode="r", key="df")
        fname2=glob.glob(hdf5_base%int(storm_yymonth_str))[0]
        o2=pd.read_hdf(fname2, mode="r", key="df")
        o=pd.concat([o1,o2])
    else:
        fname=glob.glob(hdf5_base%int(start_yymonth_str))[0]
        o=pd.read_hdf(fname, mode="r", key="df")            
        
    ###################################
    mask = (o['DATE'] > day_start_inspect) & (o['DATE'] <= day_end_inspect)
    data=o.loc[mask]
    
    ######### ONSET ################
    
    onset=data.loc[data['SYM-H']>0]
    onsetdates=onset.DATE.apply(lambda x: x.strftime("%d %H:%M"))
    if np.size(onset.to_numpy()):
        onsetmaxpos=onset.loc[onset['SYM-H'].idxmax()]
        onsetmaxtime=str(onsetmaxpos.DATE.day)+' '+str(onsetmaxpos.DATE.hour)+':'+str(onsetmaxpos.DATE.minute)
        onsetispos=True
    else:
        onsetispos=False
        print "NO POSITIVE SYM-H TO BE SHOWN AS ONSET"
    #####
    minind=data.loc[data['SYM-H'].idxmin()]
    stringminind=str(minind.DATE.day)+' '+str(minind.DATE.hour)+':'+str(minind.DATE.minute)
    
    if onsetispos:    
        print ("Onset Time: %s"%onsetdates.to_list()[0])
        print ("MAX ONSET time: %s"%onsetmaxtime)
    print ("Min Sym-H Time: %s"%stringminind)
    print ("Min Sym-H =",minind['SYM-H'])
    
    ############ PLOT ###############
    
    f=plt.figure()
    ax=f.add_subplot(111)
    ax.plot(data['DATE'],data['SYM-H'],lw=2.0,c='r')
    ax.axhline(y=0,c='k',lw=2.0)
    ax.axvline(x=minind['DATE'],c='b',lw=3.0)
    if onsetispos:
        ax.axvline(x=onsetmaxpos['DATE'],c='g',lw=3.0)
#        ax.axvline(x=onset[onset.index==onset.index[0]]['DATE'],c='r',lw=3.0)
    plt.show()
    
#    print(data[(data['DATE']>dt.datetime(yy,mm,dd,4,0,0))&(data['DATE']<dt.datetime(yy,mm,dd,5,0,0))])


def get_onset_time_new_set():
    dates=new_set_stormdays()
    hdf5_base="/home/sdch10/DATA/geomag/symh/hdf5/%d*.h5"
    proxy_onset_times=[]
    min_symh_times=[]
    for date in dates:
        yy=date.year
        mm=date.month
        dd=date.day
        thisdate=dt.datetime(yy,mm,dd,0,0,0)
        day_start_inspect=thisdate+dt.timedelta(days=-1)
        day_end_inspect=thisdate+dt.timedelta(days=1)
        
        ##### READ SYM-H FILES ##############
        start_month=day_start_inspect.month
        end_month=day_end_inspect.month
        storm_month=mm
        
        start_yymonth_str=str(yy)+'{:02d}'.format(start_month)
        end_yymonth_str=str(yy)+'{:02d}'.format(end_month)
        storm_yymonth_str=str(yy)+'{:02d}'.format(storm_month)
        ########
        if start_month==storm_month and not storm_month==end_month:
            label=1
        elif storm_month==end_month and not start_month==storm_month:
            label=2
        else:
            label=3
        ########    
        if label==1:
            fname1=glob.glob(hdf5_base%int(start_yymonth_str))[0]
            o1=pd.read_hdf(fname1, mode="r", key="df")
            fname2=glob.glob(hdf5_base%int(end_yymonth_str))[0]
            o2=pd.read_hdf(fname2, mode="r", key="df")
            o=pd.concat([o1,o2])
            o.reset_index(drop=True,inplace=True)
        elif label==2:
            fname1=glob.glob(hdf5_base%int(start_yymonth_str))[0]
            o1=pd.read_hdf(fname1, mode="r", key="df")
            fname2=glob.glob(hdf5_base%int(storm_yymonth_str))[0]
            o2=pd.read_hdf(fname2, mode="r", key="df")
            o=pd.concat([o1,o2])
            o.reset_index(drop=True,inplace=True)
        else:
            fname=glob.glob(hdf5_base%int(start_yymonth_str))[0]
            o=pd.read_hdf(fname, mode="r", key="df")
        ########
        mask = (o['DATE'] >= day_start_inspect) & (o['DATE'] <= day_end_inspect)
        data=o.loc[mask]
        minloc=data['SYM-H'].idxmin()
        isthiszeroloc=minloc
        while data.loc[isthiszeroloc]['SYM-H']<0:
            isthiszeroloc=isthiszeroloc-1
            
        zero=data.loc[isthiszeroloc]
        zero_crossing_dt=dt.datetime(zero.DATE.year,zero.DATE.month,zero.DATE.day,zero.DATE.hour,zero.DATE.minute,zero.DATE.second)
        mini=data.loc[minloc]
        min_symh_dt=dt.datetime(mini.DATE.year,mini.DATE.month,mini.DATE.day,mini.DATE.hour,mini.DATE.minute,mini.DATE.second)
        proxy_onset_times.append(zero_crossing_dt)
        min_symh_times.append(min_symh_dt)
        
    return proxy_onset_times,min_symh_times


def new_set_stormdays():
    ### gives days corresponding to observed min sym-h for the storms
    dates=[dt.datetime(2000,2,12),dt.datetime(2000,4,7),dt.datetime(2000,9,18),
           dt.datetime(2000,10,29),dt.datetime(2001,3,20),dt.datetime(2001,3,31),
           dt.datetime(2001,4,11),dt.datetime(2001,4,18),dt.datetime(2001,8,17),
           dt.datetime(2001,9,26),dt.datetime(2001,10,21),dt.datetime(2001,10,28),
           dt.datetime(2002,3,24),dt.datetime(2002,4,18),dt.datetime(2002,8,2),
           dt.datetime(2002,9,4),dt.datetime(2003,8,18),
           dt.datetime(2004,2,11),dt.datetime(2004,3,10),dt.datetime(2004,4,4),
           dt.datetime(2004,8,30),dt.datetime(2005,8,24),dt.datetime(2005,8,31),
           dt.datetime(2005,9,11),dt.datetime(2006,4,14),dt.datetime(2011,8,6),
           dt.datetime(2011,9,26),dt.datetime(2011,10,25),dt.datetime(2012,3,9),
           dt.datetime(2012,4,24),dt.datetime(2012,10,1),dt.datetime(2013,3,17),
           dt.datetime(2014,2,19),dt.datetime(2014,2,27),dt.datetime(2015,3,17),
           dt.datetime(2016,10,13),dt.datetime(2018,8,26)]
    
    return dates

    
def quiet_days_new_set():
    quietdays=[[18,19,4,17,20],[26,14,22,18,25],[10,14,11,9,22],
               [20,8,21,9,6],[15,16,26,11,17],[15,16,26,11,17],
               [30,27,24,19,25],[30,27,24,19,25],[16,24,15,11,29],
               [10,7,9,1,6],[24,18,7,17,26],[24,18,7,17,26],
               [17,28,14,16,27],[8,9,26,5,25],[6,24,7,5,25],
               [23,25,24,29,20],[31,5,16,27,4],
               [26,17,8,20,10],[24,7,6,8,25],[2,12,22,20,29],
               [4,8,24,3,25],[11,20,30,12,28],[11,20,30,12,28],
               [24,21,20,25,8],[30,12,1,3,2],[31,18,19,3,21],
               [23,19,1,16,8],[14,22,23,28,29],[26,29,25,31,20],
               [30,6,9,8,16],[20,4,29,30,22],[8,7,26,25,13],
               [13,26,14,25,2],[13,26,14,25,2],[10,30,5,14,9],
               [21,11,20,9,22],[6,14,10,13,23]]
    
    return quietdays
               
               
def gen_average_storm():
    ### Average Storm 
    ## 
#    onsets,minsyms=gen_storm_arrays()
    onsets,minsyms=get_onset_time_new_set()
    
    delh=[]
    symprofile=[]
    #############################
    ### THE FEATURES
    ### #########################
    minsymh=[]
    onsethour=[]
    falloffslope=[]
    fallofftime=[]
    rangesymh=[]
    
    for starttime,mintime in zip(onsets,minsyms):
        ###
        ###go back 24 hours
        start=np.datetime64(starttime)
        windowstart=starttime+dt.timedelta(days=-1)
        ### fast forward 48 hours
        windowend=starttime+dt.timedelta(days=2)
        ##############################################################
        ### GET DATA for window start time ###########################
        ##############################################################
        if windowstart.month==windowend.month:
            fname=getfilename(windowstart)
            o=pd.read_hdf(fname, mode="r", key="df")
            ######
            mask = (o['DATE'] > windowstart) & (o['DATE'] <= windowend)
            data=o.loc[mask]
            symh=data['SYM-H'].to_numpy()
            timestamps=data['DATE'].to_numpy()
            timeoffsets=np.array([(tstamp-start)/np.timedelta64(1,'h') for tstamp in timestamps ])
        else:
            fname1=getfilename(windowstart)
            fname2=getfilename(windowend)
            o1=pd.read_hdf(fname1, mode="r", key="df")
            o2=pd.read_hdf(fname2, mode="r", key="df")
            o=pd.concat([o1,o2])
            #####
            mask = (o['DATE'] > windowstart) & (o['DATE'] <= windowend)
            data=o.loc[mask]
            symh=data['SYM-H'].to_numpy()
            timestamps=data['DATE'].to_numpy()
            timeoffsets=np.array([(tstamp-start)/np.timedelta64(1,'h') for tstamp in timestamps ])
        
        falloff=(mintime-starttime).total_seconds()/3600.
        minsymh.append(np.min(symh))
        onsethour.append(starttime.hour+starttime.minute/60.)
        fallofftime.append(falloff)
        rangesymh.append(0.-np.min(symh))
        falloffslope.append((0.-np.min(symh))/falloff)
        #### RECORD DATA
        delh.append(timeoffsets)
        symprofile.append(symh)
    
    delh=np.array(delh)
    symprofile=np.array(symprofile)        
    avgstorm=symprofile.mean(0)
    medstorm=np.median(symprofile,axis=0)
    upperlim=np.max(symprofile,axis=0)
    lowerlim=np.min(symprofile,axis=0)
    
    
#    f=plt.figure()
#    ax=f.add_subplot(111)
#    ax.plot(delh[0],medstorm,'r-o')
#    ax.fill_between(delh[0],lowerlim,upperlim,color='c',alpha=0.5)
#    ax.set_xlabel("Hours after onset", fontsize=35)
#    ax.set_ylabel("SYM-H (nT)",fontsize=35)
#    ax.set_xlim([-24,48])
#    for tick in ax.xaxis.get_major_ticks():
#        tick.label.set_fontsize(25)    
#    for tick in ax.yaxis.get_major_ticks():
#        tick.label.set_fontsize(25)
#    plt.show()    
    return onsethour,fallofftime,falloffslope,minsymh,rangesymh,symprofile

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''' DOWNLOAD STORM TIME TEC FILES '''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def download_TEC_storm(): 
    #constants
    madrigalUrl = 'http://cedar.openmadrigal.org'
#    madrigalUrl = 'http://millstonehill.haystack.mit.edu'
#    instrument = 'World-wide GPS Receiver Network'
    instrument = 'World-wide GNSS Receiver Network'
    
    user_fullname = 'Shantanab Debchoudhury'
    user_email = 'sdch10@vt.edu'
    user_affiliation = 'Virginia Tech'
    
    # create the main object to get all needed info from Madrigal
    madrigalObj = md.MadrigalData(madrigalUrl)
    # these next few lines convert instrument name to code
    code = None
    instList = madrigalObj.getAllInstruments()
    for inst in instList:
        if inst.name.lower() == instrument.lower():
            code = inst.code
            print "Found instrument!"
            print code
            break
    
    if code == None:
        raise ValueError, 'Unknown instrument %s' % (instrument)
    
    onsets,minsyms=get_onset_time_new_set()
    for start in onsets:
        yy=start.year
        mm=start.month
        dd=start.day
        ##
        end=start+dt.timedelta(days=1)
        nextyy=end.year
        nextmm=end.month
        nextdd=end.day
        ###
        print "Now fetching file for:"
        print str(yy) + "-" + str(mm) + "-" + str(dd)
        print str(nextyy) + "-" + str(nextmm) + "-" + str(nextdd)
        ########
        expList = madrigalObj.getExperiments(code,yy,mm,dd,0,0,0,nextyy,nextmm,nextdd,0,0,0)
        for exp in expList:
            print exp.startday,'-',exp.endday
            if (exp.startday==dd or exp.startday==nextdd):
                print "Here's the experiment!"
#                print (str(exp) + '\n')    
                fileList = madrigalObj.getExperimentFiles(exp.id)
                for thisFile in fileList:
                    if thisFile.category == 1 and str(thisFile.name.split('/')[-1])[0:3]=='gps':
                        print (str(thisFile.name) + '\n')
                        thisFilename = thisFile.name
    
                        onlyFileName = "Storm_"+str(exp.startyear) + "_" + str(exp.startmonth) + "_" + str(exp.startday)+".hdf5"
#                        f = open("/home/sdch10/Datafiles/Storm_TEC_pred/" + onlyFileName[len(onlyFileName)-1],"w")
#                        f.close()
                        print "Beginning download for:"
                        print str(exp.startyear) + "-" + str(exp.startmonth) + "-" + str(exp.startday)
                        madrigalObj.downloadFile(thisFilename, "/home/sdch10/Datafiles/TEC_Files_Storm_Study/Storm_Days/" + onlyFileName, user_fullname, user_email, user_affiliation, "hdf5")
                        print "Completed download for:"
                        print str(exp.startyear) + "-" + str(exp.startmonth) + "-" + str(exp.startday)      
            else:
                print "Not this file",exp.startday,'-',exp.endday


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''' DOWNLOAD QUIET TIME TEC FILES '''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def download_TEC_quiet():
    madrigalUrl = 'http://cedar.openmadrigal.org'
#    madrigalUrl = 'http://millstonehill.haystack.mit.edu'
    instrument = 'World-wide GNSS Receiver Network'
    
    user_fullname = 'Shantanab Debchoudhury'
    user_email = 'sdch10@vt.edu'
    user_affiliation = 'Virginia Tech'
    
    # create the main object to get all needed info from Madrigal
    madrigalObj = md.MadrigalData(madrigalUrl)
    # these next few lines convert instrument name to code
    code = None
    instList = madrigalObj.getAllInstruments()
    for inst in instList:
        if inst.name.lower() == instrument.lower():
            code = inst.code
            print "Found instrument!"
            print code
            break
    
    if code == None:
        raise ValueError, 'Unknown instrument %s' % (instrument)
    
    onsets,minsyms=get_onset_time_new_set()
    quiet=quiet_days_new_set()
    for qc,start in zip(quiet,onsets):
        yy=start.year
        mm=start.month
        stormday=start.day
        
        
        for dd in qc:
            thisday=dt.datetime(yy,mm,dd)
            end=thisday+dt.timedelta(days=1)
            nextyy=end.year
            nextmm=end.month
            nextdd=end.day
            print "Now fetching file for:"
            print str(yy) + "-" + str(mm) + "-" + str(dd)
            expList = madrigalObj.getExperiments(code,yy,mm,dd,0,0,0,nextyy,nextmm,nextdd,0,0,0)
            for exp in expList:
                print exp.startday,'-',exp.endday
                if (exp.startday==dd):
                    print "Here's the experiment!"
#                    print (str(exp) + '\n')    
                    fileList = madrigalObj.getExperimentFiles(exp.id)
                    for thisFile in fileList:
                        if thisFile.category == 1 and str(thisFile.name.split('/')[-1])[0:3]=='gps':
                            print (str(thisFile.name) + '\n')
                            thisFilename = thisFile.name
        
                            onlyFileName = "Quiet_"+str(yy) + "_" + str(mm) + "_" + str(stormday)+ "_" +str(dd)+".hdf5"
    #                        f = open("/home/sdch10/Datafiles/Storm_TEC_pred/" + onlyFileName[len(onlyFileName)-1],"w")
    #                        f.close()
                            print "Beginning download for:"
                            print str(exp.startyear) + "-" + str(exp.startmonth) + "-" + str(exp.startday)
                            madrigalObj.downloadFile(thisFilename, "/home/sdch10/Datafiles/TEC_Files_Storm_Study/Quiet_Days/" + onlyFileName, user_fullname, user_email, user_affiliation, "hdf5")
                            print "Completed download for:"
                            print str(exp.startyear) + "-" + str(exp.startmonth) + "-" + str(exp.startday)      
                else:
                    print "Not this file",exp.startday,'-',exp.endday            

        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''' EXTRACT DATA FROM TEC FILES'''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def extract_data(string,start,quietday):
    yy=start.year
    mm=start.month
    dd=start.day
    ##
    end=start+dt.timedelta(days=1)
    nextyy=end.year
    nextmm=end.month
    nextdd=end.day
    if string=='storm':
        hdf5_base='/home/sdch10/Datafiles/TEC_Files_Storm_Study/Storm_Days/Storm_%d_%d_%d.hdf5'
        fname=glob.glob(hdf5_base%(yy,mm,dd))[0]
        f=hf.File(fname,'r')
        data1=f['Data']['Table Layout']
        data1=np.array(data1)
        f.close()
        fname2=glob.glob(hdf5_base%(nextyy,nextmm,nextdd))[0]
        f2=hf.File(fname2,'r')
        data2=f2['Data']['Table Layout']
        data2=np.array(data2)
        f2.close()       
        
        ### get times as datetimes
        timearray1=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data1])
        lat1=np.array([item['gdlat'] for item in data1])
        lon1=np.array([item['glon'] for item in data1])
        TEC1=np.array([item['tec'] for item in data1])
#        alt1=np.array([item['gdalt'] for item in data1])
        
        timearray2=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data2])
        lat2=np.array([item['gdlat'] for item in data2])
        lon2=np.array([item['glon'] for item in data2])
        TEC2=np.array([item['tec'] for item in data2])
#        alt2=np.array([item['gdalt'] for item in data2])
        
        timearray=np.append(timearray1,timearray2)
        lat=np.append(lat1,lat2)
        lon=np.append(lon1,lon2)
        TEC=np.append(TEC1,TEC2)
        alt=350.*np.ones(len(TEC))
            
    else:
        hdf5_base='/home/sdch10/Datafiles/TEC_Files_Storm_Study/Quiet_Days/Quiet_%d_%d_%d_%d.hdf5'
        fname=glob.glob(hdf5_base%(yy,mm,dd,quietday))[0]
        f=hf.File(fname,'r')
        data1=f['Data']['Table Layout']
        data1=np.array(data1)
        f.close()
        
        ### get times as datetimes
        timearray=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data1])
        lat=np.array([item['gdlat'] for item in data1])
        lon=np.array([item['glon'] for item in data1])
        TEC=np.array([item['tec'] for item in data1])
        alt=350.*np.ones(len(TEC))
    
    return timearray,lat,lon,TEC,alt          

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' MAKE TEC GRID SECTORS BY DIP AND DECLINATION '''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def make_magnetic_grids(date,lats,lons,alts,TECs,times):
    year=date.year
    lat0=20
    lat1=50
    lon0=-125
    lon1=-60
    
    mask=(lats>=lat0)&(lats<=lat1)&(lons<=lon1)&(lons>=lon0)
    latitude=lats[mask]
    longitude=lons[mask]
    altitude=alts[mask]
    tec=TECs[mask]
    t=times[mask]
    
    declarr=[]
    inclarr=[]
    
    for lati,loni,alti in zip(latitude,longitude,altitude):
        ret=igrf.igrf_value(lati,loni,alti,year=year)
        declarr.append(ret[0])
        inclarr.append(ret[1])
#    decl=declarr.compute()
#    incl=inclarr.compute()    
    decl=np.array(declarr)
    incl=np.array(inclarr)
    
    mask_HE=(decl<0)&(incl>65)
    mask_ME=(decl<0)&(incl>55)&(incl<=65)
    mask_LE=(decl<0)&(incl<=55)
    
    mask_HW=(decl>=0)&(incl>65)
    mask_MW=(decl>=0)&(incl>55)&(incl<=65)
    mask_LW=(decl>=0)&(incl<=55)
    
#    if string=='HE':
#        return latitude[mask_HE],longitude[mask_HE],altitude[mask_HE],tec[mask_HE],t[mask_HE]
#    elif string=='ME':
#        return latitude[mask_ME],longitude[mask_ME],altitude[mask_ME],tec[mask_ME],t[mask_ME]
#    elif string=='LE':
#        return latitude[mask_LE],longitude[mask_LE],altitude[mask_LE],tec[mask_LE],t[mask_LE]
#    elif string=='HW':
#        return latitude[mask_HW],longitude[mask_HW],altitude[mask_HW],tec[mask_HW],t[mask_HW]
#    elif string=='MW':
#        return latitude[mask_MW],longitude[mask_MW],altitude[mask_MW],tec[mask_MW],t[mask_MW]
#    elif string=='LW':
#        return latitude[mask_LW],longitude[mask_LW],altitude[mask_LW],tec[mask_LW],t[mask_HE]   

    latgrid=[latitude[mask_HE],latitude[mask_ME],latitude[mask_LE],latitude[mask_HW],latitude[mask_MW],latitude[mask_LW]]
    longgrid=[longitude[mask_HE],longitude[mask_ME],longitude[mask_LE],longitude[mask_HW],longitude[mask_MW],longitude[mask_LW]]
    altgrid=[altitude[mask_HE],altitude[mask_ME],altitude[mask_LE],altitude[mask_HW],altitude[mask_MW],altitude[mask_LW]]
    tecgrid=[tec[mask_HE],tec[mask_ME],tec[mask_LE],tec[mask_HW],tec[mask_MW],tec[mask_LW]]
    timegrid=[t[mask_HE],t[mask_ME],t[mask_LE],t[mask_HW],t[mask_MW],t[mask_LW]]

    return latgrid,longgrid,altgrid,tecgrid,timegrid
    

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PROCESS HDF5 TEC FILES TO OBTAIN STORM DATA'''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def storm_process_TEC_US():
    onsets,minsyms=get_onset_time_new_set()
    storm_hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Storm_Days/Storm_%d_%s.h5'
    domains=['HE','ME','LE','HW','MW','LW']
    for start,index in zip(onsets,range(len(onsets))):
        ##############################################
        #########  STORM DATA FIRST ##################
        ##############################################
        print("GETTING STORM DATA FOR Storm %d on %d/%d/%d"%(index,start.year,start.month,start.day))
        windowend=start+dt.timedelta(days=1)
        timearray,lat,lon,TEC,alt=extract_data('storm',start,0)
        mask=(timearray>start)&(timearray<windowend)
        
        lats=lat[mask]
        lons=lon[mask]
        TECs=TEC[mask]
        alts=alt[mask]
        times=timearray[mask]
        
        latgrid,longgrid,altgrid,tecgrid,timegrid=make_magnetic_grids(start,lats,lons,alts,TECs,times)
        for loc in range(6):
            domain=domains[loc]
            timestamps=timegrid[loc]
            tec=tecgrid[loc]
            uniquetimes,counts=np.unique(timestamps,return_counts=True)
            cum=np.cumsum(counts)
            cumul=np.append(np.array([0]),cum)
            TECarray=np.array([])
            for i in range(1,len(cumul)):
                avgTEC=np.mean(tec[cumul[i-1]:cumul[i]])
                TECarray=np.append(TECarray,avgTEC)
        ####### SAVE TO FILE
            print("Writing to file for storm %d in %s"%(index,domain))
            fname_storm=storm_hdf5_base%(index,domain)
            _df_storm = pd.DataFrame({'time': uniquetimes, 'TEC': TECarray})
            _df_storm.to_hdf(fname_storm, mode="w", key="df")
            print("Completed! Find file at %s"%(fname_storm))
        ###########################
        print("YAYY! Completed:Storm %d on %d/%d/%d"%(index,start.year,start.month,start.day))


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PROCESS HDF5 TEC FILES TO OBTAIN QUIET DATA'''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def quiet_process_TEC_US():
    onsets,minsyms=get_onset_time_new_set()
    quiet=quiet_days_new_set()
    quiet_hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Quiet_Days/Quiet_%d_%s.h5'
    #################################################        
    ################## QUIET ########################
    #################################################
    domains=['HE','ME','LE','HW','MW','LW']
    for qt,start,index in zip(quiet,onsets,range(len(onsets))):
        print("GETTING QUIET DATA FOR Storm %d on %d/%d/%d"%(index,start.year,start.month,start.day))
        df_HE=[]
        df_ME=[]
        df_LE=[]
        df_HW=[]
        df_MW=[]
        df_LW=[]
        ### GET data for each quiet day
        for q in qt:
            print("Getting data for Quiet Day %d"%(q))
            times,lats,lons,TECs,alts=extract_data('quiet',start,q)
            latgrid,longgrid,altgrid,tecgrid,timegrid=make_magnetic_grids(start,lats,lons,alts,TECs,times)
            ### ASSIGN to each sector
            for loc in range(6):
                timestamps=timegrid[loc]
                tec=tecgrid[loc]
                uniquetimes,counts=np.unique(timestamps,return_counts=True)
                cum=np.cumsum(counts)
                cumul=np.append(np.array([0]),cum)
                TECarray=np.array([])
                for i in range(1,len(cumul)):
                    avgTEC=np.mean(tec[cumul[i-1]:cumul[i]])
                    TECarray=np.append(TECarray,avgTEC)
                if loc==0:
                    df_HE.append(TECarray)
                elif loc==1:
                    df_ME.append(TECarray)
                elif loc==2:
                    df_LE.append(TECarray)                
                elif loc==3:
                    df_HW.append(TECarray)                    
                elif loc==4:
                    df_MW.append(TECarray)                
                elif loc==5:
                    df_LW.append(TECarray)        
            
        
        fname_quiet_HE=quiet_hdf5_base%(index,'HE')
        fname_quiet_ME=quiet_hdf5_base%(index,'ME')
        fname_quiet_LE=quiet_hdf5_base%(index,'LE')
        fname_quiet_HW=quiet_hdf5_base%(index,'HW')
        fname_quiet_MW=quiet_hdf5_base%(index,'MW')
        fname_quiet_LW=quiet_hdf5_base%(index,'LW')
        
        
        ### WRITE TO FILE
        print("Writing to file for storm %d in all domains"%(index))
        _df_quiet_HE = pd.DataFrame({'time': uniquetimes, 'TEC1':df_HE[0],
                                     'TEC2':df_HE[1],'TEC3':df_HE[2],
                                     'TEC4':df_HE[3],'TEC5':df_HE[4],
                                     'TEC':np.mean(df_HE,axis=0)})
        _df_quiet_HE.to_hdf(fname_quiet_HE, mode="w", key="df")
        ############
        _df_quiet_ME = pd.DataFrame({'time': uniquetimes, 'TEC1':df_ME[0],
                                     'TEC2':df_ME[1],'TEC3':df_ME[2],
                                     'TEC4':df_ME[3],'TEC5':df_ME[4],
                                     'TEC':np.mean(df_ME,axis=0)})
        _df_quiet_ME.to_hdf(fname_quiet_ME, mode="w", key="df")        
        #################
        _df_quiet_LE = pd.DataFrame({'time': uniquetimes, 'TEC1':df_LE[0],
                                     'TEC2':df_LE[1],'TEC3':df_LE[2],
                                     'TEC4':df_LE[3],'TEC5':df_LE[4],
                                     'TEC':np.mean(df_LE,axis=0)})
        _df_quiet_LE.to_hdf(fname_quiet_LE, mode="w", key="df")                
         #################   
        _df_quiet_HW = pd.DataFrame({'time': uniquetimes, 'TEC1':df_HW[0],
                                     'TEC2':df_HW[1],'TEC3':df_HW[2],
                                     'TEC4':df_HW[3],'TEC5':df_HW[4],
                                     'TEC':np.mean(df_HW,axis=0)})
        _df_quiet_HW.to_hdf(fname_quiet_HW, mode="w", key="df")
         ############  
        _df_quiet_MW = pd.DataFrame({'time': uniquetimes, 'TEC1':df_MW[0],
                                     'TEC2':df_MW[1],'TEC3':df_MW[2],
                                     'TEC4':df_MW[3],'TEC5':df_MW[4],
                                     'TEC':np.mean(df_MW,axis=0)})
        _df_quiet_MW.to_hdf(fname_quiet_MW, mode="w", key="df")
        ##############   
        _df_quiet_LW = pd.DataFrame({'time': uniquetimes, 'TEC1':df_LW[0],
                                     'TEC2':df_LW[1],'TEC3':df_LW[2],
                                     'TEC4':df_LW[3],'TEC5':df_LW[4],
                                     'TEC':np.mean(df_LW,axis=0)})
        _df_quiet_LW.to_hdf(fname_quiet_LW, mode="w", key="df") 
        print("Writing to file completed for %s\n%s\n%s\n%s\n%s\n%s\n"
              %(fname_quiet_HE,fname_quiet_ME,fname_quiet_LE,fname_quiet_HW,
                fname_quiet_MW,fname_quiet_LW))
        print("SHHHHH! 'QUIET'-LY Completed:Storm %d on %d/%d/%d"%(index,start.year,start.month,start.day))
        print("-----------------------------")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''EXTRACT STORM DATA FOR PRECONDITIONING'''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def extract_precondition_data(start):
    nextyy=start.year
    nextmm=start.month
    nextdd=start.day
    ##
    ## end is actually day before...NEVER MIND
    end=start-dt.timedelta(hours=2)
    yy=end.year
    mm=end.month
    dd=end.day
    
    hdf5_base='/home/sdch10/Datafiles/TEC_Files_Storm_Study/Storm_Days/Storm_%d_%d_%d.hdf5'
    
    if not dd==nextdd:
        fname=glob.glob(hdf5_base%(yy,mm,dd))[0]
        f=hf.File(fname,'r')
        data1=f['Data']['Table Layout']
        data1=np.array(data1)
        f.close()
        fname2=glob.glob(hdf5_base%(nextyy,nextmm,nextdd))[0]
        f2=hf.File(fname2,'r')
        data2=f2['Data']['Table Layout']
        data2=np.array(data2)
        f2.close()               
    ### get times as datetimes
        timearray1=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data1])
        lat1=np.array([item['gdlat'] for item in data1])
        lon1=np.array([item['glon'] for item in data1])
        TEC1=np.array([item['tec'] for item in data1])
    #        alt1=np.array([item['gdalt'] for item in data1])        
        timearray2=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data2])
        lat2=np.array([item['gdlat'] for item in data2])
        lon2=np.array([item['glon'] for item in data2])
        TEC2=np.array([item['tec'] for item in data2])
    #        alt2=np.array([item['gdalt'] for item in data2])
        timearray=np.append(timearray1,timearray2)
        lat=np.append(lat1,lat2)
        lon=np.append(lon1,lon2)
        TEC=np.append(TEC1,TEC2)
        alt=350.*np.ones(len(TEC))
    
    else:
        fname=glob.glob(hdf5_base%(yy,mm,dd))[0]
        f=hf.File(fname,'r')
        data1=f['Data']['Table Layout']
        data1=np.array(data1)
        f.close()
        
        ### get times as datetimes
        timearray=np.array([dt.datetime(int(item['year']),int(item['month']),int(item['day']),int(item['hour']),int(item['min']),int(item['sec'])) for item in data1])
        lat=np.array([item['gdlat'] for item in data1])
        lon=np.array([item['glon'] for item in data1])
        TEC=np.array([item['tec'] for item in data1])
        alt=350.*np.ones(len(TEC))
    
    return timearray,lat,lon,TEC,alt   
                    
""""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''GENERATE PRECONDITIONING QUIET AND STORM TEC PARAMS'''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""""""""
def preconditiong_TEC():
    ###### get ionosphere history ############
    quiet_hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Quiet_Days/Quiet_%d_%s.h5'
    onsets,minsyms=get_onset_time_new_set()
    domains=['HE','ME','LE','HW','MW','LW']
    storm_TEC_precondition=[]
    quiet_TEC_precondition=[]
    for start,index in zip(onsets,range(len(onsets))):
        ##############################################
        #########  STORM DATA FIRST ##################
        ##############################################
        print("GETTING PRECONDITIONING STORM DATA FOR Storm %d on %d/%d/%d"%(index,start.year,start.month,start.day))
        windowstart=start-dt.timedelta(hours=2)
        timearray,lat,lon,TEC,alt=extract_precondition_data(start)
        mask=(timearray>=windowstart)&(timearray<start)
        
        lats=lat[mask]
        lons=lon[mask]
        TECs=TEC[mask]
        alts=alt[mask]
        times=timearray[mask]
        
        latgrid,longgrid,altgrid,tecgrid,timegrid=make_magnetic_grids(start,lats,lons,alts,TECs,times)
        precondstorm=[]
        precondquiet=[]
        for domain,loc in zip(domains,range(6)):
            ############################################
            ### GET Storm precondition #################
            timestamps=timegrid[loc]
            tec=tecgrid[loc]
            uniquetimes,counts=np.unique(timestamps,return_counts=True)
            cum=np.cumsum(counts)
            cumul=np.append(np.array([0]),cum)
            TECarray=np.array([])
            for i in range(1,len(cumul)):
                avgTEC=np.mean(tec[cumul[i-1]:cumul[i]])
                TECarray=np.append(TECarray,avgTEC)
            
            precondstorm.append(np.mean(TECarray))
            
            ########################################
            ### Now quiet preconditioning #########
            quiet_fname=quiet_hdf5_base%(index,domain)            
            quietdata=pd.read_hdf(quiet_fname,mode="r", key="df")
            
            stime=np.array([pt.strftime('%H:%M:%S') for pt in uniquetimes])   
            
            qTEC=quietdata.TEC.to_numpy()
            qtime=quietdata.time.apply(lambda x: x.strftime('%H:%M:%S')).to_numpy()
            
            pivot=np.where(qtime==stime[0])[0]                
            quietTEC=np.roll(qTEC,-pivot[0])            
            quiettime=np.roll(qtime,-pivot[0])
                     
            precondquiet.append(np.mean(quietTEC[0:len(stime)]))                            
            ############################################
        print("GOT Preconditioning TEC parameters for this storm")    
            
        storm_TEC_precondition.append(precondstorm)
        quiet_TEC_precondition.append(precondquiet)

        
    return storm_TEC_precondition,quiet_TEC_precondition    
        
""""'''''''''''''''''''''''''''''''''''''''''''''''''
'''''COMPUTE delta TEC for each storm '''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''"""""""""              
def get_delta_TEC():
    storm_hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Storm_Days/Storm_%d_%s.h5'
    quiet_hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Quiet_Days/Quiet_%d_%s.h5'
    
    onsets,minsyms=get_onset_time_new_set()
    domains=['HE','ME','LE','HW','MW','LW']
    ##########
    df_HE=[]
    df_ME=[]
    df_LE=[]
    df_HW=[]
    df_MW=[]
    df_LW=[]
    ##########
    for start,index in zip(onsets,range(len(onsets))):     
        for domain in domains:
            storm_fname=storm_hdf5_base%(index,domain)
            quiet_fname=quiet_hdf5_base%(index,domain)
            
            stormdata=pd.read_hdf(storm_fname,mode="r", key="df")
            quietdata=pd.read_hdf(quiet_fname,mode="r", key="df")
            
            sTEC=stormdata.TEC.to_numpy()
            stime=stormdata.time.apply(lambda x: x.strftime('%H:%M:%S')).to_numpy()
            
            qTEC=quietdata.TEC.to_numpy()
            qtime=quietdata.time.apply(lambda x: x.strftime('%H:%M:%S')).to_numpy()
            
            if True in np.isnan(qTEC):
                locnan=np.argwhere(np.isnan(qTEC))
                for l2 in locnan:
                    l=l2[0]
                    qTEC[l]=(qTEC[l-2]+qTEC[l-1]+qTEC[l+1]+qTEC[l+2])/4.0
                
            
            pivot=np.where(qtime==stime[0])[0]
            if not np.array_equal(stime,np.roll(qtime,-pivot[0])):
                print ("RED FLAG!! ABORT")
                
            quietTEC=np.roll(qTEC,-pivot[0])        
            
            if domain=='HE':
                df_HE.append(sTEC)
                df_HE.append(quietTEC)
            elif domain=='ME':
                df_ME.append(sTEC)
                df_ME.append(quietTEC)
            elif domain=='LE':
                df_LE.append(sTEC)
                df_LE.append(quietTEC)                
            elif domain=='HW':
                df_HW.append(sTEC)
                df_HW.append(quietTEC)                    
            elif domain=='MW':
                df_MW.append(sTEC)
                df_MW.append(quietTEC)               
            elif domain=='LW':
                df_LW.append(sTEC)
                df_LW.append(quietTEC)
            
    return df_HE,df_ME,df_LE,df_HW,df_MW,df_LW

""""'''''''''''''''''''''''''''''''''''''''''''''''''
'''''WRITE input output in hdf5 FILES''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''""""""""" 
def write_to_file_params():
    #####################################
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    onsethour_float=[]
    for start in ons:
        onsethour_float.append(round(start.hour+start.minute/60.,2))   
    onsethour_float=np.array(onsethour_float)
    onsine=np.sin(2 * np.pi * onsethour_float/24.0)
    oncos=np.cos(2 * np.pi * onsethour_float/24.0)
    storm_pre,quiet_pre=preconditiong_TEC()
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Datafile_%s.h5'
    domains=['HE','ME','LE','HW','MW','LW']
    ####################################################
    ##### ONE FILE WITH ALL PARAMETERS #################
    for domain,index in zip(domains,range(6)):
        if domain=='HE':
            data=df_HE
        elif domain=='ME':
            data=df_ME
        elif domain=='LE':
            data=df_LE              
        elif domain=='HW':
            data=df_HW                 
        elif domain=='MW':
            data=df_MW            
        elif domain=='LW':
            data=df_LW
            
        storm_TEC=data[::2]
        quiet_TEC=data[1::2]
        
        ############################################
        ####### Y variables ########################
        dTEC=np.array(storm_TEC)-np.array(quiet_TEC)
        dTECmean=np.mean(dTEC,axis=1)    
        dTECstd=np.std(dTEC,axis=1)
        dTECabs=np.mean(abs(dTEC),axis=1)
        
        
        #############################################
        ###### PRECONDITION VARIABLES ###############
        pre_storm=np.array([])
        pre_quiet=np.array([])
        
        for s_chunk,q_chunk in zip(storm_pre,quiet_pre):
            pre_storm=np.append(pre_storm,s_chunk[index])
            pre_quiet=np.append(pre_quiet,q_chunk[index])
        
        pre_delta=pre_storm-pre_quiet
        
        
        ######### SYM PRECONDITION ####################
        ### GET MEAN SYM 2hrs prior######################
        storm_sym_precond=symprofile[:,1440-120:1440]
        mean_sym_history=np.mean(storm_sym_precond,axis=1)
        ######### Mean storm sym #######################
        storm_sym=symprofile[:,1440:1440*2]
        mean_sym_storm=np.mean(storm_sym,axis=1)
        ###############################################
        
        ############## LOCAL TIME #####################
        if domain[1]=='E':
            localtimes=onsethour_float-4.0
        else:
            localtimes=onsethour_float-7.0
            
        locsine=np.sin(2 * np.pi * localtimes/24.0)
        loccos=np.cos(2 * np.pi * localtimes/24.0)        
        
        _feat=pd.DataFrame({'Storm History':pre_storm,'Quiet History':pre_quiet,'Delta History':pre_delta,
                       'Fallofftime':np.array(fallofftime),'Slope':np.array(slope),'Minimum SYM':np.array(minsymh),
                       'Range SYM':rangesymh,'Mean SYM':mean_sym_storm,'SYM History':mean_sym_history,
                       'Onset UT':onsethour_float,'UT Onset sine':onsine,'UT Onset cos':oncos,
                       'Onset LT':localtimes,'LT Onset sine':locsine,'LT Onset cos':loccos,
                       'Average dTEC':dTECmean,'Std dTEC':dTECstd,'Absolute dTEC':dTECabs})
    
        fname=hdf5_base%domain
        print("Writing to file for %s"%domain)
        _feat.to_hdf(fname,mode="w", key="df")
        print("Completed feature listing")
        print("---------------")


def main_phase_only_write_to_file():
    #####################################
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    onsethour_float=[]
    for start in ons:
        onsethour_float.append(round(start.hour+start.minute/60.,2))   
    onsethour_float=np.array(onsethour_float)
    onsine=np.sin(2 * np.pi * onsethour_float/24.0)
    oncos=np.cos(2 * np.pi * onsethour_float/24.0)
    mains=[]
    for on_dt,min_dt in zip(ons,mins):
        mains.append(int((min_dt-on_dt).total_seconds()/300.))
    
    storm_pre,quiet_pre=preconditiong_TEC()
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Mainphase_Datafile_%s.h5'
    domains=['HE','ME','LE','HW','MW','LW']
    ####################################################
    ##### ONE FILE WITH ALL PARAMETERS #################
    for domain,index in zip(domains,range(6)):
        if domain=='HE':
            data=df_HE
        elif domain=='ME':
            data=df_ME
        elif domain=='LE':
            data=df_LE              
        elif domain=='HW':
            data=df_HW                 
        elif domain=='MW':
            data=df_MW            
        elif domain=='LW':
            data=df_LW
            
        storm_TEC=data[::2]
        quiet_TEC=data[1::2]
        
        ############################################
        ####### Y variables ########################
        dTEC=np.array(storm_TEC)-np.array(quiet_TEC)
        dTECmean=np.array([])
        dTECstd=np.array([])
        dTECabs=np.array([])
        for r in range(np.shape(storm_TEC)[0]):
            dTECmean=np.append(dTECmean,np.mean(dTEC[r,0:mains[r]+1]))    
            dTECstd=np.append(dTECstd,np.std(dTEC[r,0:mains[r]+1])) 
            dTECabs=np.append(dTECabs,np.mean(abs(dTEC[r,0:mains[r]+1]))) 
            print("Storm Num=%d,mainphase in %d"%(r,mains[r]))
        
        #############################################
        ###### PRECONDITION VARIABLES ###############
        pre_storm=np.array([])
        pre_quiet=np.array([])
        
        for s_chunk,q_chunk in zip(storm_pre,quiet_pre):
            pre_storm=np.append(pre_storm,s_chunk[index])
            pre_quiet=np.append(pre_quiet,q_chunk[index])
        
        pre_delta=pre_storm-pre_quiet
        
        
        ######### SYM PRECONDITION ####################
        ### GET MEAN SYM 2hrs prior######################
        storm_sym_precond=symprofile[:,1440-120:1440]
        mean_sym_history=np.mean(storm_sym_precond,axis=1)
        ######### Mean storm sym #######################
        storm_sym=symprofile[:,1440:1440*2]
        mean_sym_storm=np.array([])
        for s in range(np.shape(storm_sym)[0]):
            mean_sym_storm=np.append(mean_sym_storm,np.mean(storm_sym[s,0:mains[s]+1]))
        ###############################################
        
        ############## LOCAL TIME #####################
        if domain[1]=='E':
            localtimes=onsethour_float-4.0
        else:
            localtimes=onsethour_float-7.0
            
        locsine=np.sin(2 * np.pi * localtimes/24.0)
        loccos=np.cos(2 * np.pi * localtimes/24.0)        
        
        _feat=pd.DataFrame({'Storm History':pre_storm,'Quiet History':pre_quiet,'Delta History':pre_delta,
                       'Fallofftime':np.array(fallofftime),'Slope':np.array(slope),'Minimum SYM':np.array(minsymh),
                       'Range SYM':rangesymh,'Mean SYM':mean_sym_storm,'SYM History':mean_sym_history,
                       'Onset UT':onsethour_float,'UT Onset sine':onsine,'UT Onset cos':oncos,
                       'Onset LT':localtimes,'LT Onset sine':locsine,'LT Onset cos':loccos,
                       'Average dTEC':dTECmean,'Std dTEC':dTECstd,'Absolute dTEC':dTECabs})
    
        fname=hdf5_base%domain
        print("Writing to file for %s"%domain)
        _feat.to_hdf(fname,mode="w", key="df")
        print("Completed feature listing")
        print("---------------")
    
    


""""'''''''''''''''''''''''''''''''''''''''''''''''''
'''''GET PLOTS OF TEC for each storm '''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''""""""""" 
def plot_TEC_individual_storm(num):
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Mainphase_Datafile_%s.h5'
    domains=['HW','HE','MW','ME','LW','LE']
    shists=np.array([])
    qhists=np.array([])
    meansyms=np.array([])      
    for domain,i in zip(domains,range(6)):
        fname=hdf5_base%domain
        data=pd.read_hdf(fname,mode='r',key='df')
        X=data.drop(labels=["Average dTEC","Std dTEC","Absolute dTEC","Onset UT",
                            "Onset LT","LT Onset sine","LT Onset cos","Range SYM"],axis=1)       
        y=data['Average dTEC']
        shists=np.append(shists,X['Storm History'][num])
        qhists=np.append(qhists,X['Quiet History'][num])
        meansyms=np.append(meansyms,X['Mean SYM'][num])
    
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    ######
    xaxis=np.arange(0,288,1)/12.
    storm_TEC_HE=df_HE[::2]
    quiet_TEC_HE=df_HE[1::2]
    storm_TEC_ME=df_ME[::2]
    quiet_TEC_ME=df_ME[1::2]
    storm_TEC_LE=df_LE[::2]
    quiet_TEC_LE=df_LE[1::2]
    storm_TEC_HW=df_HW[::2]
    quiet_TEC_HW=df_HW[1::2]
    storm_TEC_MW=df_MW[::2]
    quiet_TEC_MW=df_MW[1::2]
    storm_TEC_LW=df_LW[::2]
    quiet_TEC_LW=df_LW[1::2]      
    ######
    UTonset=np.round(onsethour[num],2)
    mainphase=np.round(fallofftime[num],2)
    storm_time_sym=symprofile[:,24*60:48*60]
    thisstormsym=storm_time_sym[num]
    ######
    f=plt.figure(figsize=(25,15))
    ax=f.add_subplot(111)
    ax.plot(xaxis,storm_TEC_HE[num]-quiet_TEC_HE[num],'r-o',lw=3.0,label="Upper East")
    ax.plot(xaxis,storm_TEC_ME[num]-quiet_TEC_ME[num],'b-o',lw=3.0,label="Middle East")    
    ax.plot(xaxis,storm_TEC_LE[num]-quiet_TEC_LE[num],'g-o',lw=3.0,label="Lower East")
    ax.plot(xaxis,storm_TEC_HW[num]-quiet_TEC_HW[num],'c-o',lw=3.0,label="Upper West")       
    ax.plot(xaxis,storm_TEC_MW[num]-quiet_TEC_MW[num],'m-o',lw=3.0,label="Middle West")
    ax.plot(xaxis,storm_TEC_LW[num]-quiet_TEC_LW[num],'k-o',lw=3.0,label="Lower West")       
    ax.set_xlabel("Hours from onset",fontsize=25)
    ax.set_ylabel("Total Electron Content (TECu)",fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.legend(prop={'size':25})
    ax.set_xlim([0,24])
#    ax.set_ylim([-35,35])
    ax.set_title("Storm on %d/%d/%d, starting at %.2f UT"%(ons[num].month,ons[num].day,
                                                         ons[num].year,
                                                         UTonset),fontsize=25)
    ax.axvline(x=mainphase,lw=2.0,color='k',alpha=0.3)
    ax2=ax.twinx()
    ax2.plot(xaxis,thisstormsym[::5],'y--',alpha=0.5,lw=2.0)
    ax2.set_ylabel('SYM-H (nT)',fontsize=25)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)
    print shists,qhists,meansyms
    
""""'''''''''''''''''''''''''''''''''''''''''''''''''
'''''GET PLOTS OF TEC for all storms '''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''""""""""" 
def plot_TEC_all_storm():
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    ######
    xaxis=np.arange(0,288,1)/12.
    storm_TEC_HE=df_HE[::2]
    quiet_TEC_HE=df_HE[1::2]
    storm_TEC_ME=df_ME[::2]
    quiet_TEC_ME=df_ME[1::2]
    storm_TEC_LE=df_LE[::2]
    quiet_TEC_LE=df_LE[1::2]
    storm_TEC_HW=df_HW[::2]
    quiet_TEC_HW=df_HW[1::2]
    storm_TEC_MW=df_MW[::2]
    quiet_TEC_MW=df_MW[1::2]
    storm_TEC_LW=df_LW[::2]
    quiet_TEC_LW=df_LW[1::2]      
    ######
    f=plt.figure(figsize=(25,15))
    ax=f.add_subplot(111)
    ax.plot(xaxis,np.mean(storm_TEC_HE,axis=0)-np.mean(quiet_TEC_HE,axis=0),'r-o',lw=3.0,label="Upper East")
    ax.plot(xaxis,np.mean(storm_TEC_ME,axis=0)-np.mean(quiet_TEC_ME,axis=0),'b-o',lw=3.0,label="Middle East")    
    ax.plot(xaxis,np.mean(storm_TEC_LE,axis=0)-np.mean(quiet_TEC_LE,axis=0),'g-o',lw=3.0,label="Lower East")
    ax.plot(xaxis,np.mean(storm_TEC_HW,axis=0)-np.mean(quiet_TEC_HW,axis=0),'c-o',lw=3.0,label="Upper West")       
    ax.plot(xaxis,np.mean(storm_TEC_MW,axis=0)-np.mean(quiet_TEC_MW,axis=0),'m-o',lw=3.0,label="Middle West")
    ax.plot(xaxis,np.mean(storm_TEC_LW,axis=0)-np.mean(quiet_TEC_LW,axis=0),'k-o',lw=3.0,label="Lower West")       
    ax.set_xlabel("Hours from onset",fontsize=25)
    ax.set_ylabel("$\Delta$TEC (TECu)",fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.legend(prop={'size':25})
    ax.set_xlim([0,24])
    ax.axhline(y=0,c='k',lw=2.0,alpha=0.5)
#    ax.set_ylim([-35,35])
    
"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''CORRELATION OF PARAMETERS''''''''''''''''''''''
"""''''''''''''''''''''''''''''''''''''''''''''''''''""""""    
def plot_correlation_importance():
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Datafile_%s.h5'
    domains=['HW','HE','MW','ME','LW','LE']
    fig, axs = plt.subplots(3,2, figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top=0.915,bottom=0.18,left=0.04,right=0.98,
                        hspace=0.14,wspace=0.075 )
    axs=axs.ravel()
    for domain,i in zip(domains,range(6)):
        fname=hdf5_base%domain
        data=pd.read_hdf(fname,mode='r',key='df')
        X=data.drop(labels=["Average dTEC","Std dTEC","Absolute dTEC","Onset UT",
                            "Onset LT","LT Onset sine","LT Onset cos","Range SYM"],axis=1)       
        y=data['Average dTEC']
#        y=data['Std dTEC']
        corr_array=[abs(spear(X[k],y)[0]) for k in X.keys()]
        ##### PLOT ##############
        index = np.arange(len(corr_array))+0.5
        bar_width = 0.8
        opacity = 0.8
        axs[i].bar(index,corr_array,bar_width, alpha=opacity, color='xkcd:red')
        axs[i].set_ylim([0,0.7])
        if i>3:
            axs[i].set_xlabel('Storm Characteristics',labelpad=0.4,fontsize=20)
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            axs[i].set_xticks(index-0.5)
            axs[i].set_xticklabels(X.keys())
            axs[i].tick_params(axis='x', which='both',length=0)
            for tick in axs[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
                tick.label.set_rotation(45)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
        else:
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            empty_string_labels = ['']*len(X.keys())
            axs[i].set_xticklabels(empty_string_labels)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)            
        axs[i].set_title(domain,fontsize=15)  
        axs[i].grid(axis='y')
        plt.suptitle("Unsigned Spearman Rank Correlation with std of $\Delta$ TEC",fontsize=25)

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''PARTIAL CORRELATION OF PARAMETERS''''''''''''''''''''''
"""''''''''''''''''''''''''''''''''''''''''''''''''''""""""    
def plot_partial_correlation_importance():
    var='Std'
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Datafile_%s.h5'
    domains=['HW','HE','MW','ME','LW','LE']
    fig, axs = plt.subplots(3,2, figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top=0.915,bottom=0.18,left=0.04,right=0.98,
                        hspace=0.14,wspace=0.075 )
    axs=axs.ravel()
    for domain,i in zip(domains,range(6)):
        fname=hdf5_base%domain
        data=pd.read_hdf(fname,mode='r',key='df')

        if var=='Average':
            X=data.drop(labels=["Std dTEC","Absolute dTEC","Onset UT",
                            "Onset LT","LT Onset sine","LT Onset cos","Range SYM"],axis=1)
            pcorr=partial_corr(X)
            corr_array=abs(np.delete(pcorr[0],0))
            keys=np.delete(X.keys(),0)
        else:
            X=data.drop(labels=["Average dTEC","Absolute dTEC","Onset UT",
                            "Onset LT","LT Onset sine","LT Onset cos","Range SYM"],axis=1)
            pcorr=partial_corr(X)
            corr_array=abs(np.delete(pcorr[7],7))
            keys=np.delete(X.keys(),7)
            
        ##### PLOT ##############
        index = np.arange(len(corr_array))+0.5
        bar_width = 0.8
        opacity = 0.8
        axs[i].bar(index,corr_array,bar_width, alpha=opacity, color='xkcd:red')
        axs[i].set_ylim([0,0.5])
        if i>3:
            axs[i].set_xlabel('Storm Characteristics',labelpad=0.4,fontsize=20)
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            axs[i].set_xticks(index-0.5)
            axs[i].set_xticklabels(keys)
            axs[i].tick_params(axis='x', which='both',length=0)
            for tick in axs[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
                tick.label.set_rotation(45)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
        else:
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            empty_string_labels = ['']*len(keys)
            axs[i].set_xticklabels(empty_string_labels)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)            
        axs[i].set_title(domain,fontsize=15)  
        axs[i].grid(axis='y')
        plt.suptitle("Partial Correlation with %s of $\Delta$ TEC"%var,fontsize=25)



def plot_correlation_importance_mainphase():
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Mainphase_Datafile_%s.h5'
    domains=['HW','HE','MW','ME','LW','LE']
    fig, axs = plt.subplots(3,2, figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top=0.915,bottom=0.18,left=0.04,right=0.98,
                        hspace=0.14,wspace=0.075 )
    axs=axs.ravel()
    
    for domain,i in zip(domains,range(6)):
        fname=hdf5_base%domain
        data=pd.read_hdf(fname,mode='r',key='df')
        X=data.drop(labels=["Average dTEC","Std dTEC","Absolute dTEC","Onset UT",
                            "Onset LT","LT Onset sine","LT Onset cos","Range SYM"],axis=1)       
        y=data['Average dTEC']
        mask=X['Fallofftime']>16
        X=X[mask]
        y=y[mask]
#        y=data['Std dTEC']
        corr_array=[abs(spear(X[k],y)[0]) for k in X.keys()]
        ##### PLOT ##############
        index = np.arange(len(corr_array))+0.5
        bar_width = 0.8
        opacity = 0.8
        axs[i].bar(index,corr_array,bar_width, alpha=opacity, color='xkcd:red')
        axs[i].set_ylim([0,0.7])
        if i>3:
            axs[i].set_xlabel('Storm Characteristics',labelpad=0.4,fontsize=20)
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            axs[i].set_xticks(index-0.5)
            axs[i].set_xticklabels(X.keys())
            axs[i].tick_params(axis='x', which='both',length=0)
            for tick in axs[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
                tick.label.set_rotation(45)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
        else:
#            axs[i].set_ylabel('Unsigned spearman coefficient')
            empty_string_labels = ['']*len(X.keys())
            axs[i].set_xticklabels(empty_string_labels)
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)            
        axs[i].set_title(domain,fontsize=15)  
        axs[i].grid(axis='y')
        plt.suptitle("Unsigned Spearman Rank Correlation with MAINPHASE avg of $\Delta$ TEC",fontsize=25)

def clock_plot(domain):
    hdf5_base='/home/sdch10/Datafiles/TEC_Processed_Storm_Study/Feature_Study/Datafile_%s.h5'
    fname=hdf5_base%domain
    data=pd.read_hdf(fname,mode='r',key='df')
    f=plt.figure()
    ax=f.add_subplot(121)
    cmap=matplotlib.cm.RdBu_r
    ff=ax.scatter(data['UT Onset sine'],data['UT Onset cos'],c=data['Average dTEC'],s=500,cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=1.00)
    cbar=f.colorbar(ff,cax=cax,orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.set_xlabel('Average $\Delta$TEC (TECu)',fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.set_aspect('equal')
    ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_xlabel('Sine of 24-hr UT clock',fontsize=25)
    ax.set_ylabel('Cos of 24-hr UT clock',fontsize=25)
    ax.annotate(s='', xy=(0.8,0), xytext=(-0.75,0), arrowprops=dict(facecolor='black',arrowstyle='<->'),fontsize=25)
    ax.annotate(s='', xy=(0,-0.8), xytext=(0,0.8), arrowprops=dict(facecolor='black',arrowstyle='<->'),fontsize=25)
    ax.text(x=-0.04,y=-0.8,s='12',fontsize=25)
    ax.text(x=0.8,y=-0.04,s='6',fontsize=25)
    ax.text(x=-0.9,y=-0.04,s='18',fontsize=25)
    ax.text(x=-0.04,y=0.9,s='0',fontsize=25)
    ax.grid(which='major',axis='both')
    ax.invert_yaxis()
    ##########
    ax2=f.add_subplot(122)
    ff2=ax2.scatter(data['UT Onset sine'],data['UT Onset cos'],c=data['Std dTEC'],s=500,cmap=cmap)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="5%", pad=1.00)
    cbar2=f.colorbar(ff2,cax=cax2,orientation="horizontal")
    cax2.xaxis.set_ticks_position("bottom")
    cbar2.ax.tick_params(labelsize=25)
    cbar2.ax.set_xlabel('Std $\Delta$TEC (TECu)',fontsize=25)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_aspect('equal')
    ax2.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
    ax2.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax2.set_xlabel('Sine of 24-hr UT clock',fontsize=25)
    ax2.set_ylabel('Cos of 24-hr UT clock',fontsize=25)
    ax2.annotate(s='', xy=(0.8,0), xytext=(-0.75,0), arrowprops=dict(facecolor='black',arrowstyle='<->'),fontsize=25)
    ax2.annotate(s='', xy=(0,-0.8), xytext=(0,0.8), arrowprops=dict(facecolor='black',arrowstyle='<->'),fontsize=25)
    ax2.text(x=-0.04,y=-0.8,s='12',fontsize=25)
    ax2.text(x=0.8,y=-0.04,s='6',fontsize=25)
    ax2.text(x=-0.9,y=-0.04,s='18',fontsize=25)
    ax2.text(x=-0.04,y=0.9,s='0',fontsize=25)
    ax2.grid(which='major',axis='both')
    ax2.invert_yaxis()
    ##############
    f.subplots_adjust(top=0.915,
    bottom=0.095,
    left=0.06,
    right=1.0,
    hspace=0.2,
    wspace=0.075)
    f.suptitle('Onset UT dependence on post-storm $\Delta$TEC distribution in High Latitude Positive Declination',fontsize=25)
        

def sunrise_plot(domain):
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    onsethour_float=[]
    for start in ons:
        onsethour_float.append(round(start.hour+start.minute/60.,2))   
    onsethour_float=np.array(onsethour_float)
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    if domain=='HE':
        data=df_HE
    elif domain=='ME':
        data=df_ME
    elif domain=='LE':
        data=df_LE              
    elif domain=='HW':
        data=df_HW                 
    elif domain=='MW':
        data=df_MW            
    elif domain=='LW':
        data=df_LW
    
    storm_TEC=data[::2]
    quiet_TEC=data[1::2]
    
    ############################################
    dTEC=np.array(storm_TEC)-np.array(quiet_TEC)
    matrix=np.zeros([24,288])
    countmat=np.zeros(24)
    nstorms=np.shape(storm_TEC)[0]
    for i in range(nstorms):
        thisonset=int(onsethour_float[i])
        countmat[thisonset]+=1
        matrix[thisonset]=matrix[thisonset]+dTEC[i]
    
    for count,i in zip(countmat,range(24)):
        if not count==0:
            matrix[i]=matrix[i]/count
    #############
    mymin=np.min(matrix[np.nonzero(matrix)])
    mymax=np.max(matrix[np.nonzero(matrix)])
    
    yaxis=np.arange(0,24,0.1)
    xaxis_sunrise=12.*np.ones(np.size(yaxis))-yaxis
    xaxis_sunrise=xaxis_sunrise%24
    locs=np.where(xaxis_sunrise>12)[0]
    demarc=locs[0]
    print countmat
    ##############
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cmap=matplotlib.cm.RdBu_r
    ret=ax.pcolor(matrix,cmap=cmap,vmin=mymin,vmax=mymax,norm=MidpointNormalize(mymin, mymax, 0.))
    cbar=fig.colorbar(ret,ax=ax)
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.set_ylabel('$\Delta$TEC (TECu)',fontsize=25)
    ax.set_yticks(np.arange(24)+0.5)
    ax.set_yticklabels(np.arange(24))
    ax.set_xticks(np.arange(0,288,24))
    ax.set_xticklabels(np.arange(0,288,24)/12)
    ax.set_xlabel("Hours after onset",fontsize=25)
    ax.set_ylabel("Onset Time (UT)",fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.plot(xaxis_sunrise[0:demarc]*12,yaxis[0:demarc],'k--',lw=2.0)
    ax.plot(xaxis_sunrise[demarc:]*12,yaxis[demarc:],'k--',lw=2.0)
    plt.title('Effect of Onset Time',fontsize=25)
    plt.show()
        
    
def sunrise_plot_mainphase_sorted(domain,dur):
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    onsethour_float=[]
    for start in ons:
        onsethour_float.append(round(start.hour+start.minute/60.,2))   
    onsethour_float=np.array(onsethour_float)
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    if domain=='HE':
        data=df_HE
    elif domain=='ME':
        data=df_ME
    elif domain=='LE':
        data=df_LE              
    elif domain=='HW':
        data=df_HW                 
    elif domain=='MW':
        data=df_MW            
    elif domain=='LW':
        data=df_LW
    
    storm_TEC=data[::2]
    quiet_TEC=data[1::2]
    
    ############################################
    dTEC=np.array(storm_TEC)-np.array(quiet_TEC)
    matrix=np.zeros([24,288])
    countmat=np.zeros(24)
    nstorms=np.shape(storm_TEC)[0]
    
    for i in range(nstorms):
        thisonset=int(onsethour_float[i])
        thisfalloff=fallofftime[i]
        if thisfalloff<dur:
            countmat[thisonset]+=1
            matrix[thisonset]=matrix[thisonset]+dTEC[i]
    
    for count,i in zip(countmat,range(24)):
        if not count==0:
            matrix[i]=matrix[i]/count
    #############
    mymin=np.min(matrix[np.nonzero(matrix)])
    mymax=np.max(matrix[np.nonzero(matrix)])
    
    yaxis=np.arange(0,24,0.1)
    xaxis_sunrise=12.*np.ones(np.size(yaxis))-yaxis
    xaxis_sunrise=xaxis_sunrise%24
    locs=np.where(xaxis_sunrise>12)[0]
    demarc=locs[0]
#    print countmat
    ##############
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cmap=matplotlib.cm.RdBu_r
    ret=ax.pcolor(matrix,cmap=cmap,vmin=mymin,vmax=mymax,norm=MidpointNormalize(mymin, mymax, 0.))
    cbar=fig.colorbar(ret,ax=ax)
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.set_ylabel('$\Delta$TEC (TECu)',fontsize=25)
    ax.set_yticks(np.arange(24)+0.5)
    ax.set_yticklabels(np.arange(24))
    ax.set_xticks(np.arange(0,288,24))
    ax.set_xticklabels(np.arange(0,288,24)/12)
    ax.set_xlabel("Hours after onset",fontsize=25)
    ax.set_ylabel("Onset Time (UT)",fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.plot(xaxis_sunrise[0:demarc]*12,yaxis[0:demarc],'k--',lw=2.0)
    ax.plot(xaxis_sunrise[demarc:]*12,yaxis[demarc:],'k--',lw=2.0)
    plt.title('Effect of Onset Time for storms <%d hr mainphase'%int(dur),fontsize=25)
    plt.show()    
    
def sunrise_dependence_compute(domain,data,ons,numhrs):
    '''
    numhrs: how many hours after sunrise? negative is before sunrise
    domain: which sector
    '''
    onsethour_float=[]
    for start in ons:
        onsethour_float.append(round(start.hour+start.minute/60.,2))   
    onsethour_float=np.array(onsethour_float)
    a=Astral()
    a.solar_depression = 'civil'
    
    if domain=='HE':
        city=a['New York']
    elif domain=='ME':
        city=a['New York']
    elif domain=='LE':
        city=a['New York']              
    elif domain=='HW':
        city=a['Phoenix']                 
    elif domain=='MW':
        city=a['Phoenix']            
    elif domain=='LW':
        city=a['Phoenix']    

    storm_TEC=data[::2]
    quiet_TEC=data[1::2]
    
    ############################################
    dTEC=np.array(storm_TEC)-np.array(quiet_TEC)

    nums=len(onsethour_float)
    avgsunrise=np.zeros(nums)
    startsunrise=[]
    for i in range(nums):
        dTECarr=dTEC[i,:]
        #######
        sun=city.sun(date=ons[i],local=False)
        sunrisehr=sun['sunrise'].hour
        sunrisemin=sun['sunrise'].minute
        sunrisetime=sunrisehr+sunrisemin/60.
        #######
        timetosun=(sunrisetime-onsethour_float[i])%24
        startsunrise.append(timetosun)
        #######

        if numhrs>0:
            start=int(timetosun*12)
            if timetosun<(24-numhrs):
                end=start+numhrs*12
            else:
                end=288
            mean_dTEC=np.mean(dTECarr[start:end])
            avgsunrise[i]=mean_dTEC
        else:
            end=int(timetosun*12)
            if timetosun>abs(numhrs):
                start=end+numhrs*12
            else:
                start=0
            mean_dTEC=np.mean(dTECarr[start:end])
            avgsunrise[i]=mean_dTEC

    
    return avgsunrise,np.array(startsunrise)

def sunrise_effect_by_sector():
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    domains=['HW','HE','MW','ME','LW','LE']
    offsethrs=[-3,-2,-1,1,2,3,4]
    sunrise_dTEC=[]
    stds=[]
    for domain,i in zip(domains,range(6)):
        if domain=='HE':
            data=df_HE
        elif domain=='ME':
            data=df_ME
        elif domain=='LE':
            data=df_LE             
        elif domain=='HW':
            data=df_HW               
        elif domain=='MW':
            data=df_MW
        elif domain=='LW':
            data=df_LW
        
        avgsunrise=[]
        stdsunrise=[]
        for offhr in offsethrs:
            avgeffect,sunrises=sunrise_dependence_compute(domain,data,ons,offhr)
            avgsunrise.append(np.mean(avgeffect))
            stdsunrise.append(np.std(avgeffect))
        sunrise_dTEC.append(avgsunrise)
        stds.append(stdsunrise)
        
        ################
    f=plt.figure(figsize=(25,15))
    ax=f.add_subplot(111)
    ax.plot(offsethrs,sunrise_dTEC[0],'r-o',lw=4.0,ms=8,label=domains[0])
    ax.plot(offsethrs,sunrise_dTEC[1],'g-o',lw=4.0,ms=8,label=domains[1])    
    ax.plot(offsethrs,sunrise_dTEC[2],'b-o',lw=4.0,ms=8,label=domains[2])    
    ax.plot(offsethrs,sunrise_dTEC[3],'c-o',lw=4.0,ms=8,label=domains[3])    
    ax.plot(offsethrs,sunrise_dTEC[4],'m-o',lw=4.0,ms=8,label=domains[4])    
    ax.plot(offsethrs,sunrise_dTEC[5],'k-o',lw=4.0,ms=8,label=domains[5])   
    
#    ax.errorbar(offsethrs,sunrise_dTEC[0],yerr=stds[0],ecolor='r',elinewidth=3,capthick=5)
#    ax.errorbar(offsethrs,sunrise_dTEC[1],yerr=stds[1],ecolor='g',elinewidth=3,capthick=5)   
#    ax.errorbar(offsethrs,sunrise_dTEC[2],yerr=stds[2],ecolor='b',elinewidth=3,capthick=5)    
#    ax.errorbar(offsethrs,sunrise_dTEC[3],yerr=stds[3],ecolor='c',elinewidth=3,capthick=5)    
#    ax.errorbar(offsethrs,sunrise_dTEC[4],yerr=stds[4],ecolor='m',elinewidth=3,capthick=5)    
#    ax.errorbar(offsethrs,sunrise_dTEC[5],yerr=stds[5],ecolor='k',elinewidth=3,capthick=5) 
    
    ax.set_xlabel('Hours offset from sunrise',fontsize=25)
    ax.set_ylabel('Average $\Delta$TEC for all storms (TECu)',fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)    
    ax.legend(prop={'size':25})
#    plt.title("Unsigned Spearman Rank Correlation with MAINPHASE avg of $\Delta$ TEC",fontsize=25)

def histogram_sunrise_effect_by_sector():
    onsethour,fallofftime,slope,minsymh,rangesymh,symprofile=gen_average_storm()
    ons,mins=get_onset_time_new_set()
    
    df_HE,df_ME,df_LE,df_HW,df_MW,df_LW=get_delta_TEC()
    domains=['HW','HE','MW','ME','LW','LE']
    fig, axs = plt.subplots(3,2, figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top=0.914,
                        bottom=0.077,
                        left=0.064,
                        right=0.99,
                        hspace=0.294,
                        wspace=0.108 )
    bins=np.linspace(-10,10,25)
    axs=axs.ravel()    
    for domain,i in zip(domains,range(6)):
        if domain=='HE':
            data=df_HE
        elif domain=='ME':
            data=df_ME
        elif domain=='LE':
            data=df_LE             
        elif domain=='HW':
            data=df_HW               
        elif domain=='MW':
            data=df_MW
        elif domain=='LW':
            data=df_LW
            
        avgeffect_2,sunrises_2=sunrise_dependence_compute(domain,data,ons,-2)
        avgeffect_1,sunrises_1=sunrise_dependence_compute(domain,data,ons,-1)
        avgeffect1,sunrises1=sunrise_dependence_compute(domain,data,ons,1)
        avgeffect2,sunrises2=sunrise_dependence_compute(domain,data,ons,2)
        
        m, s = stats.norm.fit(avgeffect_2) # get mean and standard deviation  
        pdf = norm.pdf(bins, m, s) # now get theoretical values in our interval  
        axs[i].plot(bins, pdf, 'b',lw=3.0, label="2 hr before")
        axs[i].axvline(x=m,c='b',ls='--',lw=3.0)
        
        m, s = stats.norm.fit(avgeffect_1) # get mean and standard deviation  
        pdf = norm.pdf(bins, m, s) # now get theoretical values in our interval  
        axs[i].plot(bins, pdf, 'c',lw=3.0,label="1 hr before")
        axs[i].axvline(x=m,c='c',ls='--',lw=3.0)
        
        m, s = stats.norm.fit(avgeffect1) # get mean and standard deviation  
        pdf = norm.pdf(bins, m, s) # now get theoretical values in our interval  
        axs[i].plot(bins, pdf, 'g',lw=3.0,label="1 hr after")
        axs[i].axvline(x=m,c='g',ls='--',lw=3.0)
        
        m, s = stats.norm.fit(avgeffect2) # get mean and standard deviation  
        pdf = norm.pdf(bins, m, s) # now get theoretical values in our interval  
        axs[i].plot(bins, pdf, 'r',lw=3.0,label="2 hr after")
        axs[i].axvline(x=m,c='r',ls='--',lw=3.0)
        
        axs[i].set_ylim([0,0.2])
#        axs[i].hist(avgeffect_2,bins,color='b',alpha=0.05,density=1,label='2 hr before')
#        axs[i].hist(avgeffect_1,bins,color='c',alpha=0.05,density=1,label='1 hr before')
#        axs[i].hist(avgeffect1,bins,color='m',alpha=0.05,density=1,label='1 hr after')
#        axs[i].hist(avgeffect2,bins,color='r',alpha=0.05,density=1,label='2 hr after')
        
                
        if i>3:
            axs[i].set_xlabel('Hours relative to sunrise',labelpad=0.4,fontsize=20)
        if i in [0,2,4]:
            axs[i].set_ylabel('Probability density',fontsize=20)
        
        axs[i].legend(prop={'size':15})
#            axs[i].set_ylabel('Unsigned spearman coefficient')
#            axs[i].set_xticks(bins)
#            axs[i].set_xticklabels(bins)
#            axs[i].tick_params(axis='x', which='both',length=0)
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)            
        axs[i].set_title(domain,fontsize=15)  
    
    plt.suptitle("Distribution of $\Delta$TEC variation relative to sunrise for all storms",fontsize=25)
#    plt.tight_layout()
        

if __name__ == "__main__":
#    on,fall,slope,minsym,rangesym,symprofile=gen_average_storm()
#    download_TEC_storm()
#    download_TEC_quiet()
    print("STARTING..IN A GALAXY FAR FAR AWAY")
#    storm_process_TEC_US()
#    quiet_process_TEC_US()   
#    write_to_file_params()
#    main_phase_only_write_to_file()
#    plot_TEC_individual_storm(2)
#    plot_TEC_all_storm()
#    plot_correlation_importance_mainphase()
#    clock_plot('HW')
    sunrise_plot('HW')
#    sunrise_plot_mainphase_sorted('HW',5)
#    gen_average_storm()
#    plot_partial_correlation_importance()
#    sunrise_effect_by_sector()
#    histogram_sunrise_effect_by_sector()