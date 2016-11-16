#from IPython.display import display # Allows the use of display() for DataFrames
import pylab as pl
from matplotlib import pyplot
from sklearn.manifold import MDS
import pandas as pd
import seaborn as sn

#Visualising data in 2 dimensions    
def VisualiseSamples(data_samples):
    print "\nReducing Samples to a 2D space"
    mds = MDS()
    MDS_Transformed_Data_Samples = mds.fit_transform(data_samples)
    MDS_Transformed_Data_Samples_df = pd.DataFrame(MDS_Transformed_Data_Samples)
    
    print "Visualising Sample Distribution"
    fig1 = pyplot.figure()
    pyplot.scatter(MDS_Transformed_Data_Samples_df.iloc[:,0], MDS_Transformed_Data_Samples_df.iloc[:,1])
    fig1.canvas.set_window_title("Sample Data Distribution")
    pyplot.show()
    print "Visualisation complete"
    
    print "Visualising feature distributions"
    fig2 = pyplot.figure()
    n, bins, patches = pyplot.hist(MDS_Transformed_Data_Samples_df.iloc[:,0])
    fig2.canvas.set_window_title("MDS Reduced Feature 1 Histogram")
    pyplot.plot(bins)
    pyplot.show()
    
    fig3 = pyplot.figure()
    n, bins, patches = pyplot.hist(MDS_Transformed_Data_Samples_df.iloc[:,1])
    fig3.canvas.set_window_title("MDS Reduced Feature 2 Histogram")
    pyplot.plot(bins)
    pyplot.show()    
    print "Visualisation Complete!"

#Plots the training and testing scores for a given model by changing it's features    
#def PlotLineGraph(plot_title, x_axis_label, y_axis_label, values, plot_lines, plot_labels):
def PlotLineGraph(plot_title, x_axis_label, y_axis_label, values, plot_line1, plot_line2, plot_label1, plot_label2):
    print "\nPlotting ", plot_title   
    fig = pl.figure()
    fig.canvas.set_window_title(plot_title)
    #for i in range(0, len(plot_lines)):
    #    pl.plot(values, plot_lines[i], lw=2, label = plot_labels[i])
    pl.plot(values, plot_line2, lw=2, label = plot_label2)
    pl.plot(values, plot_line1, lw=2, label = plot_label1)
    
    pl.legend()
    pl.xlabel(x_axis_label)
    pl.ylabel(y_axis_label)
    pl.show()
    print plot_title, " Plotted!"
    
def VisualiseHeatMaps(heat_map_data, plot_title):
    print "\nVisualising ", plot_title
    fig = pyplot.figure()
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title, fontsize=18, fontweight='bold')
    ax = pyplot.axes()

    heat_map_data_df = pd.DataFrame(heat_map_data, index=["Actual False", "Actual True"], columns=["Predicted False", "Predicted True"])
     
    sn.heatmap(heat_map_data_df, annot=True, fmt="d", linewidths=.5, ax = ax)
    pyplot.show()