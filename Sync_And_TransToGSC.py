import sys

import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import interpolate
from pyquaternion import Quaternion

from scipy.integrate import odeint
from os import listdir
from os.path import isdir, join, isfile, dirname
from scipy.signal import butter, filtfilt

def pre_w(df):
    #интерполяция угловых скоростей для численного решения уравнения пуассона
    T = df['Timestamp[nanosec]']/10**9
    W = np.asarray([df[' gx[rad/s]'],df[' gy[rad/s]'],df[' gz[rad/s]']]).T
    tck1 = interpolate.splrep(T, W[:,0], s=0)
    tck2 = interpolate.splrep(T, W[:,1], s=0)
    tck3 = interpolate.splrep(T, W[:,2], s=0)
    def w(t):
        ynew1 = interpolate.splev(t, tck1, der=0)
        ynew2 = interpolate.splev(t, tck2, der=0)
        ynew3 = interpolate.splev(t, tck3, der=0)
        return np.asarray([ynew1,ynew2,ynew3])
    return w

def create_ode(func_w):
    def poisson(Y,t):
        #система оде для уравнения пуассона
        dYdt = 0.5*Quaternion(scalar=0, vector=func_w(t))*Quaternion(Y)
        return dYdt.elements
    return poisson

def sync_and_toGSC(df1,df2,path):
    ##сначала применяется ФНЧ ко всем данным
    b, a = butter(3, 0.02)

    df1[' ax[m/s^2]'] = filtfilt(b,a,df1[' ax[m/s^2]'])
    df1[' ay[m/s^2]'] = filtfilt(b,a,df1[' ay[m/s^2]'])
    df1[' az[m/s^2]'] = filtfilt(b,a,df1[' az[m/s^2]'])

    df1[' gx[rad/s]'] = filtfilt(b,a,df1[' gx[rad/s]'])
    df1[' gy[rad/s]'] = filtfilt(b,a,df1[' gy[rad/s]'])
    df1[' gz[rad/s]'] = filtfilt(b,a,df1[' gz[rad/s]'])

    df2[' ax[m/s^2]'] = filtfilt(b,a,df2[' ax[m/s^2]'])
    df2[' ay[m/s^2]'] = filtfilt(b,a,df2[' ay[m/s^2]'])
    df2[' az[m/s^2]'] = filtfilt(b,a,df2[' az[m/s^2]'])

    df2[' gx[rad/s]'] = filtfilt(b,a,df2[' gx[rad/s]'])
    df2[' gy[rad/s]'] = filtfilt(b,a,df2[' gy[rad/s]'])
    df2[' gz[rad/s]'] = filtfilt(b,a,df2[' gz[rad/s]'])


    gir1 = np.asarray(np.sqrt(df1[' gx[rad/s]']**2+df1[' gy[rad/s]']**2+df1[' gz[rad/s]']**2))
    gir2 = np.asarray(np.sqrt(df2[' gx[rad/s]']**2+df2[' gy[rad/s]']**2+df2[' gz[rad/s]']**2))

    b, a = butter(3, 0.01)
    gir1 = filtfilt(b, a, gir1)
    b, a = butter(3, 0.01)
    gir2 = filtfilt(b, a, gir2)


    #Перебором находится время сдвига для синхронизации отдельно для акселерометра и ДУС
    maxerr = 10000000
    best_diff1 = 0
    for diff in tqdm(range(100, len(df1)+len(df2)-100)): #
        err = np.linalg.norm(gir1[max(0,len(df1)-diff):min(len(df1),len(df1)+len(df2)-diff)] - gir2[max(0,diff-len(df1)):min(diff, len(df2))])/(min(diff, len(df2))-max(0,diff-len(df1)))
        if err < maxerr:
            best_diff1 = diff
            maxerr = err


    acc1 = np.asarray(np.sqrt(df1[' ax[m/s^2]']**2+df1[' ay[m/s^2]']**2+df1[' az[m/s^2]']**2))
    acc2 = np.asarray(np.sqrt(df2[' ax[m/s^2]']**2+df2[' ay[m/s^2]']**2+df2[' az[m/s^2]']**2))

    b, a = butter(3, 0.01)
    acc1 = filtfilt(b, a, acc1)
    b, a = butter(3, 0.01)
    acc2 = filtfilt(b, a, acc2)

    maxerr = 1000000
    best_diff2 = 0
    for diff in tqdm(range(100, len(df1)+len(df2)-100)): #
        err = np.linalg.norm(acc1[max(0,len(df1)-diff):min(len(df1),len(df1)+len(df2)-diff)] - acc2[max(0,diff-len(df1)):min(diff, len(df2))])/(min(diff, len(df2))-max(0,diff-len(df1)))
        if err < maxerr:
            best_diff2 = diff
            maxerr = err

    #Если результаты совпадают - то синхронизировать удалось успешно, если нет - либо много ошибок, либо данные не имеют параллельных участков
    time_len = (df2['Timestamp[nanosec]'].max()-df2['Timestamp[nanosec]'].min())/(10**9)
    time_sync1 = (df1['Timestamp[nanosec]'][2]-df1['Timestamp[nanosec]'][1])/(10**9)*(best_diff1-len(df1))
    time_sync2 = (df1['Timestamp[nanosec]'][2]-df1['Timestamp[nanosec]'][1])/(10**9)*(best_diff2-len(df1))
    if abs(time_sync1-time_sync2)>0.8:
        print("can't sync some files. check input files")
        time_sync = np.nan
    else:
        time_sync = (time_sync1+time_sync2)/2


    w_res = []
    acc_res = []
    prest = max(0,best_diff2-len(df1))
    prefin = min(best_diff2, len(df2))
    #учёт рассинхронизации для более лучшего качества поиска общей перегрузки для приведения в общее гск
    modul = np.asarray(np.sqrt(df2[' ax[m/s^2]']**2+df2[' ay[m/s^2]']**2+df2[' az[m/s^2]']**2))[prest:prefin]
    st_acc=int(max(np.argmax(modul)-100,0))
    st=int(max(np.argmax(modul)-50,1))
    fin=int(min(len(df2),np.argmax(modul)+150))
    st_acc = st_acc + prest
    st = st + prest
    fin = fin + prest
        
    #решаем уравнение пуассона, чтобы потом привести каждое устройство в своё ГСК, совпадающие с ЛСК устройств в общий момент времени
    func_w = pre_w(df2)
    poisson = create_ode(func_w)
    res = odeint(poisson,[1,0,0,0], np.linspace(df2['Timestamp[nanosec]'].min()/10**9 + st_acc*(df2['Timestamp[nanosec]'][2]-df2['Timestamp[nanosec]'][1])/(10**9),df2['Timestamp[nanosec]'].max()/10**9,len(df2)-st_acc))
        
    #нахождение направление вектора гравитации
    j = 0
    g_gsc = []
    for el in np.asarray([df2[' ax[m/s^2]'][st_acc:st],df2[' ay[m/s^2]'][st_acc:st],df2[' az[m/s^2]'][st_acc:st]]).T:
        g_gsc.append(Quaternion(res[j]).conjugate.rotate(el))
        j = j + 1
    g_gsc = np.asarray(g_gsc)
    gz = g_gsc.T.mean(axis=1)
    ez = gz/np.linalg.norm(gz)

    #выбираем окно, которому соответствует движение с максимальным ускорением по какой-то оси
    acc_start = np.asarray([df2[' ax[m/s^2]'][st:fin],df2[' ay[m/s^2]'][st:fin],df2[' az[m/s^2]'][st:fin]]).T
    j = 0
    acc_gsc = []
    for el in acc_start:
        acc_gsc.append(Quaternion(res[j+st-st_acc]).conjugate.rotate(el))
        j = j + 1
    acc_gsc = np.asarray(acc_gsc)
    without_g = acc_gsc - gz 

    ans = without_g.T.sum(axis=1)
    ans = ans/np.linalg.norm(ans)

    #вычисляем эту ось (в идеале это ось разгона самолёта, но может быть выбрана и какая-то другая - но общая для двух устройств)
    #делаем эти оси перпендикулярными и строим по ним ГСК
    ex = ans - ans.dot(ez)*ez
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    matrix_rot = np.asarray([ex,ey,ez]).T

    #перевод измерений в ГСК
    res = odeint(poisson,[1,0,0,0], np.linspace(df2['Timestamp[nanosec]'].min()/10**9,df2['Timestamp[nanosec]'].max()/10**9,len(df2)))
    w_new = []
    acc_new = []
    mat_corr = (Quaternion(res[st_acc]).conjugate)*Quaternion(matrix=matrix_rot)
    for i in tqdm(range(len(df2))):
        acc_new.append(((Quaternion(res[i])*mat_corr).conjugate).rotate(np.asarray([df2[' ax[m/s^2]'][i],df2[' ay[m/s^2]'][i],df2[' az[m/s^2]'][i]])))
        w_new.append(((Quaternion(res[i])*mat_corr).conjugate).rotate(np.asarray([df2[' gx[rad/s]'][i],df2[' gy[rad/s]'][i],df2[' gz[rad/s]'][i]])))
    w_new = np.asarray(w_new)
    acc_new = np.asarray(acc_new)
    w_res.append(w_new)
    acc_res.append(acc_new)

    
    res_df = pd.DataFrame(np.asarray([df2['Timestamp[nanosec]'], w_new[:,0], w_new[:,1], w_new[:,2], acc_new[:,0], acc_new[:,1], acc_new[:,2]]).T)
    res_df.columns = ['Timestamp[nanosec]',' gx[rad/s]',' gy[rad/s]',' gz[rad/s]',' ax[m/s^2]',' ay[m/s^2]',' az[m/s^2]']
    res_df.set_index('Timestamp[nanosec]', inplace=True)
    res_df.to_csv(path+"\\GSC_gyro_accel.csv", encoding='utf-8')
    
    return time_sync,time_len



print('input PATH of each file or from root (Y/N)')
path = []
flag_path = (str(input())=='Y')
dfs = []
if flag_path:
    print("input directory for information about sync:")
    cur_path = str(input())
    print("input number of files:")
    n = int(input())
    for i in range(n):
        print("input path to file:")
        path_c = str(input())
        if (isfile(path_c+"\\gyro_accel.csv")):
            path.append(path_c)
            df = pd.read_csv(path[-1]+"\\gyro_accel.csv")
            dfs.append(df)
else:
    cur_path =dirname(sys.argv[0])
    print(cur_path)
    onlydirs = [d for d in listdir(cur_path) if isdir(join(cur_path, d))]
    for p in onlydirs:
        if (isfile(cur_path+'\\'+p+"\\gyro_accel.csv")):
            path.append(cur_path+'\\'+p)
            df = pd.read_csv(path[-1]+"\\gyro_accel.csv")
            dfs.append(df)


times = []
for i in range(len(dfs)-1):
    time_sync, time_len = sync_and_toGSC(dfs[i],dfs[i+1],path[i+1])
    times.append([time_sync, time_len])
_, time_len = sync_and_toGSC(dfs[-1],dfs[0],path[0])


add_inf = []
add_inf.append([path[0], time_len, '-'])
j = 0
for el in times:
    add_inf.append([path[j+1], el[1], el[0]])
    j = j + 1
add_inf = pd.DataFrame(add_inf)
add_inf.columns = ["Path to current file", "Time length of current file [sec]", "Sync time relative previous file [sec]"]
add_inf.to_csv(cur_path+"\\sync_times.csv", encoding='utf-8')
