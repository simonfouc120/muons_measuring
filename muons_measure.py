import os
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.filters import gaussian
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import seaborn as sns
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
import matplotlib.cm as cm



def cos2theta(x, a, b, c):
    return a*np.cos((x-b)/180.*np.pi)**2+c


def list_files_in_repository(directory_path):
    # List to store file paths
    file_list = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Join the root path and file name to get the full file path
            file_list.append(os.path.join(root, file))
        file_list.sort(key=lambda x: os.path.basename(x[0]).lower())
    return file_list



def density_scatter(x, y, ax=None, sort=True, bins=20, cmap='plasma', **kwargs):
    """
    Scatter plot colored by 2D histogram with density interpolation.
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), 
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    z[np.isnan(z)] = 0.0

    if sort:
        idx = np.argsort(z)  # Sort based on z values
        x, y, z = np.array(x)[idx], np.array(y)[idx], np.array(z)[idx]
    sc = ax.scatter(x, y, c=z, cmap=cmap, **kwargs)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax, fig



Files = list_files_in_repository(r"C:/Users/sf270338/MMA/Caliste-MMA/Muons cosmiques/DATA_20240912_20240922")
#
# Files += list_files_in_repository("/Users/ol161303/Documents/PROGRAMMES/IDL-DATA/prog_idl/02_STIX/PIPELINE/V_20200412/MMA/11_CALISTE_MM_CONFIG/DATA_20240830_20240919/20240917_20240919-LT30")
Files = sorted(Files)


# search crossing muons
n_error         = 0     # number of abdormal mulitplicities
n_good          = 0     # number of normal multiplicities
Multi_min       = 6     # minimal multiplicity
Multi_max       = 22    # maximal multiplicity
angle_shift     = 0     #+6    # detector image rotation angle
n_good_trace    = 0     # number of muon trace considered good
n_sigma         = 3     # gaussian filter of the hough function to detect lines
angle  = [] # muon angle from normal in degrees
date   = [] # date of trace to later find the rate
date_wrong = [] # date of wrong trace to later find the rate

for file in Files[0:10000]:
    
#    print(file)
    d = np.load(file)
    l = d.files
    keys_to_remove = {'energy_list', 'pixel_number', 'multiplicity', 'timestamp_particle'}
    l = [key for key in l if key not in keys_to_remove]
    m = np.unique(np.asarray([np.int32(l[np.int32(np.char.find(l, '_'))+1:]) for l in l ]))
    if m.size > 0 and np.max(m) > Multi_max: # select images with less than 22 pixels fired (risk of noise with LT30)
        n_error += 1 # a bad multi is detected and rejected
    elif m.size and np.max(m)> Multi_min: # select traces with more than 8 pixels fired
        n_good_trace += 1 # a good event is detected
        
        # build date from file name
        idx = np.int32(np.char.find(file, '.npz'))
        date.append(pd.to_datetime(file[idx-15:idx], format="%Y%m%d_%H%M%S"))

        # select 2D image for the current trace
        field   = 'hitmap_'+str(m[-1]) #max multiplicity in this file
        im      = d[field].reshape((16, 16))
        im      = rotate(im,90+angle_shift) # rotate the image to get the image top in imshow()

        # apply Hough transfor and filter to identify a lign into the image
        hough, theta, dist          = hough_line(im)
        hough                       = gaussian(hough, sigma=n_sigma) # smooth Hough space transform
        hough_m, theta_m, dist_m    = hough_line_peaks(hough, theta, dist)
        theta_m_abs                 = np.abs(theta_m) # search smallest angle in multiple angel hough
        idx = np.argmin(theta_m_abs) # take the minimum angle found in hough
        angle.append(theta_m[idx]/np.pi*180)   #register angle
        # take first hough angle whatever number of houg lines found
        print('...', theta_m/np.pi*180)
#        plt.imshow(im)
#        plt.show()
#        for _, agl, dist in zip(hough_m, theta_m, dist_m):
#            (x0, y0) = dist * np.array([np.cos(agl), np.sin(agl)])
#            plt.axline((x0, y0), slope=np.tan(agl + np.pi / 2))
#            print(np.tan(agl + np.pi / 2))
#            print(agl/np.pi*180)
#        plt.show()
#        if theta_m.shape[0] == 1:
#            n_good_trace +=1
#            angle.append(theta_m[0]/np.pi*180)
#        else:
#            plt.imshow(im)
#            for _, agl, dist in zip(hough_m, theta_m, dist_m):
#                (x0, y0) = dist * np.array([np.cos(agl), np.sin(agl)])
#                plt.axline((x0, y0), slope=np.tan(agl + np.pi / 2))
#                print(np.tan(agl + np.pi / 2))
#                print(agl/np.pi*180)
#            plt.show()
    else:
        n_good += 1

    #        print('.......GOOD NO TRACE.......')
    #        print(file)

# i want to select the date only from 13 september to 22 september
filtered_data = [(d, a) for d, a in zip(date, angle) if pd.to_datetime('20240913', format="%Y%m%d") < d < pd.to_datetime('20240923', format="%Y%m%d")]
date, angle = zip(*filtered_data) if filtered_data else ([], [])


print('number of issues', n_error)
print('number of goods', n_good)
print('number of good traces', n_good_trace)
print('mean angle', np.mean(angle))
n_bins = 14

#plt.hist(angle, bins=n_bins)
plt.figure('angle distribution muons')
hh, bb = np.histogram(angle, bins = n_bins)
x = (bb[0:-1]+bb[1:])/2
plt.bar(x, hh, width=np.mean(bb[0:-1]-bb[1:]), label = 'Muons angles distribution')
# fit Cos2 theta
popt, pcov = curve_fit(cos2theta, x, hh)
print('best fit parameters', popt)
print('detector tilt angle with respect to vertical %4.2f°'%(popt[1]//180)) #module 180°
plt.plot(x, cos2theta(x, *popt), 'r', label = 'best fit')
# plot setup
plt.plot([0, 0], [0, popt[0]], '--k')
plt.plot([popt[1]//180, popt[1]//180], [0, popt[0]], ':r')
plt.xlabel('Zenithal Angle [°]')
plt.ylabel('Entries')
plt.title('Caliste-MMA, cosmic-ray muons angle distribution')
plt.legend()
plt.show()


#### Muongram ####
data = pd.DataFrame({'date': date, 'angle': angle})  # Create a DataFrame with date and angle
data['day'] = data['date'].dt.date  # Extract the day from the date
angles = data['angle'].values  # Extract angle values
interval_hours = 12   # Define the interval in hours
total_days = (data['date'].max() - data['date'].min()).days + 1  # Calculate the total number of days
n_bins_x = total_days * 24 // interval_hours  # Calculate the number of bins for the x-axis
n_bins_y = int(max(angles)/2)
days = mdates.date2num(data['date'])  # Convert dates to matplotlib's date format
plt.figure('muongram')  
plt.hist2d(days, angles, bins=[n_bins_x, n_bins_y], cmap='viridis', norm=LogNorm()) 
plt.colorbar(label='Number of Entries')  
locator = mdates.DayLocator(interval=max(total_days // 10, 1))  
plt.gca().xaxis.set_major_locator(locator) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  
plt.gcf().autofmt_xdate() 
plt.xlabel('Date')  
plt.ylabel('Zenithal Angle [°]')  
plt.title(f'Caliste-MMA, cosmic-ray muons angle distribution over {interval_hours}-hour intervals')  
plt.show() 
#################

# Muongram and plot with rolling time interval windows
data = pd.DataFrame({'date': date, 'angle': angle})
data['day'] = data['date'].dt.date  
window_size = 12  # Taille de la fenêtre en heures
step_size = 1  # Taille du pas en heures
days = mdates.date2num(data['date'])
angles = data['angle'].values
total_hours = int((data['date'].max() - data['date'].min()).total_seconds() // 3600)
num_windows = total_hours - window_size + 1
window_dates = []
window_angles = []
muons_count = []
mean_windows_date = []
for start_hour in range(0, num_windows, step_size):
    end_hour = start_hour + window_size
    start_time = data['date'].min() + pd.Timedelta(hours=start_hour)
    end_time = data['date'].min() + pd.Timedelta(hours=end_hour)
    window_data = data[(data['date'] >= start_time) & (data['date'] < end_time)]
    
    window_dates.extend(mdates.date2num(window_data['date']))
    window_angles.extend(window_data['angle'])
    muons_count.append(len(window_data['angle']))
    mean_windows_date.append(mdates.date2num(window_data['date'].mean()))

n_bins_x = num_windows // step_size
n_bins_y = int(max(angles) / 2)

plt.figure('muongram')
plt.hist2d(window_dates, window_angles, bins=[n_bins_x, n_bins_y], cmap='plasma', norm=LogNorm())
plt.colorbar(label='Number of Entries')
locator = mdates.HourLocator(interval=window_size)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Zenithal Angle [°]')
plt.title(f'Caliste-MMA, cosmic-ray muons angle distribution over {window_size}-hour rolling windows')
plt.show()

plt.figure('debug')
plt.plot(mean_windows_date, muons_count)
plt.xlabel('Date')
plt.title('Muons detected vs Date')
plt.ylabel('Muons detected')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.show()
#################


#### plot the angle vs date #####
plt.figure('detected muon angle vs date')
plt.plot(date, angle, '.k')
plt.xlabel('Date [1 minute resolution]')
plt.ylabel('Detected Muon angle [°]')
plt.show()
#################################


##### Cluster on angle of muons vs date ########
date_numeric = mdates.date2num(date) 
ax,fig = density_scatter(date_numeric, angle, bins=50, cmap='plasma', alpha=0.7)
ax.set_title('Detected Muon angle vs date with density contours using interpolation')
ax.set_xlabel('Date [1 minute resolution]')
ax.set_ylabel('Detected Muon angle [°]')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.legend(['Muon angle'])
plt.show()
#############################################

# plot the number of muons detected per day
plt.figure('muons per day')
unique_days = data['day'].unique()
plt.hist(date, bins = len(unique_days)*2)
plt.xlabel('Date')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.ylabel('Number of muons detected')
plt.title('Number of muons detected per day')
plt.show()
#############################################

n_files = len(Files)
RealTime = date[-1]-date[0]
RealTime = RealTime.total_seconds() #REAL TIME
LiveTime_estimate = (n_good+n_error+n_good_trace)*60
print("LiveTime Fraction = %4.2f %%"%(LiveTime_estimate/RealTime*100.))

date = np.asarray(date)
Time_intervals  = np.asarray([(date[i+1] - date[i]).total_seconds() for i in range(len(date)-1)])
print('Livetime Corrected average muon rate %4.2f Muons/hour'%(1./np.mean(Time_intervals)*3600*RealTime/LiveTime_estimate))
print('check %4.2f Muons/hour'%(n_good_trace/LiveTime_estimate*3600.))


# plot the date to debug
plt.figure('date')
plt.plot(date)
plt.plot(date_wrong)
plt.show()



# for the record, findng corresponding histogram and data
#
## search for muons in images
## as follows, in the correct orientation
## http://51.83.70.239:10000/api/v1/caliste/id/9827?multiplicity=1
#im_ref = d['hitmap_1'].reshape((16, 16))
## rotate 90 so the Top of detector is top of image
#im_ref = rotate(im_ref, 90)
#plt.imshow(im_ref)
#
#im = d['hitmap_16'].reshape((16, 16))
#im = rotate(im, 90)
#plt.imshow(im)
#
#hough, theta, dist = hough_line(im)
#hough = gaussian(hough, sigma=0.1)
#hough_m, theta_m, dist_m = hough_line_peaks(hough, theta, dist)
#for _, angle, dist in zip(hough_m, theta_m, dist_m):
#    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#    plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))
#    print(np.tan(angle + np.pi / 2))
#    print(angle/np.pi*180)
#


