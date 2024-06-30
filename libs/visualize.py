import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def mesh_data_to_geojson(mesh_data, keys=None):
    """
    Convert mesh data to GeoJSON format.
    :param mesh_data: List of dictionaries containing mesh data.
    :param keys: List of keys to include in the properties of the GeoJSON.
    :return: A dictionary in GeoJSON format.
    """
    features = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": mesh["geometry"]},
        "properties": {k: mesh[k] for k in keys}
    } for mesh in mesh_data]

    return {"type": "FeatureCollection", "features": features}
        
class StackedFeatureGrapher:
    def __init__(self, quantization, day_counts=None):
        self.day_counts = day_counts if day_counts else [5, 2] 
        self.quantization = quantization        
        self.color_list = ["sienna", "darkorange", "yellowgreen", "seagreen", "mediumturquoise", "royalblue", "plum", "green", "orchid", "grey",
                           "lime", "azure", "lavender", "blue", "green", "beige", "navy", "violet", "skyblue"]
    
    def sort_aggregated_data(self, data_array):
        '''
        Sorting for the stacked graph
        Calculate the value for each cluster and sort in ascending order
        Assign a smaller cluster number to clusters with a strong tendency for shorter daytime stays on weekdays
        '''
        dow_weight = [0.0001, 0.0002]  # Weekday:Weekend = 1:2
        elapsed_weight = [self.quantization.e_thresholds[0]/2] + [e for e in self.quantization.e_thresholds]
        dt_weight = [(d+1)%self.quantization.dt_quant_num for d in range(self.quantization.dt_quant_num)]
        weight = np.array([[[dow_weight[dow]*elapsed_weight[e]*dt_weight[dt] for dt in range(self.quantization.dt_quant_num)] for e in range(self.quantization.e_quant_num)] for dow in range(self.quantization.dow_quant_num)])
        
        values = []  # Value for each cluster
        for cls in range(self.cluster_num):
            value = (self.elapsed_each_time_each_cluster[cls]*weight).sum()/self.elapsed_each_time_each_cluster[cls].sum()
            values.append(value)
        sorted_indices = np.argsort(values)
        return sorted_indices
    
    
    def aggregate_data_from_actual_features(self, all_df, cluster_col = "cluster", split_col = "elapsed_class", sort = False):
        self.cluster_num = len(np.unique(all_df[cluster_col]))
        self.split_num = len(np.unique(all_df[split_col]))
        self.elapsed_each_time_each_cluster = np.zeros((self.cluster_num, 2, self.split_num, self.quantization.dt_quant_num))
        def update_data(data_array, cls, is_weekend, split, dt_num, add_value):
            data_array[cls][is_weekend][split][dt_num] += add_value

        for cls, dow, sdt, e, ih, split in tqdm(zip(all_df[cluster_col], all_df["day_of_week"], all_df["arrival_time"], 
                                             all_df["stay_time"], all_df["is_holiday"], all_df[split_col])):
    
            dt_num = int(sdt.hour) * 2 + sdt.minute // 30
            stay_num = int(e // 30) + 1
            add_value = 1 / self.day_counts[1 if dow in ["Saturday", "Sunday"] or ih else 0]
            for k in range(stay_num):
                is_weekend = 1 if dow in ["Saturday", "Sunday"] or ih else 0
                update_data(self.elapsed_each_time_each_cluster, cls, is_weekend, split, dt_num, add_value)
                dt_num = (dt_num + 1) % self.quantization.dt_quant_num
        if sort:
            sorted_indices = self.sort_aggregated_data(self.elapsed_each_time_each_cluster)
            self.elapsed_each_time_each_cluster = self.elapsed_each_time_each_cluster[sorted_indices]
            return sorted_indices
            
    def aggregate_data_from_approximated_features(self, data_each_mesh, result, sort = False):
        self.cluster_num = max(result) + 1
        self.elapsed_each_time_each_cluster = np.zeros((self.cluster_num, 2, self.quantization.e_quant_num, self.quantization.dt_quant_num))
        self.mesh_count_each_cluster = np.zeros((self.cluster_num))
        for cls in range(self.cluster_num):
            cluster_mids = list(np.where(np.array(result) == cls))[0]
            self.elapsed_each_time_each_cluster[cls] = np.sum(data_each_mesh[cluster_mids], axis = 0)
            self.mesh_count_each_cluster[cls] = len(cluster_mids)
        if sort:
            sorted_indices = list(self.sort_aggregated_data(self.elapsed_each_time_each_cluster))
            self.elapsed_each_time_each_cluster = self.elapsed_each_time_each_cluster[sorted_indices]
            self.mesh_count_each_cluster = self.mesh_count_each_cluster[sorted_indices]
            return sorted_indices
        
    def get_max_value(self, data):
        """Helper method to return the maximum value of the data"""
        return np.max(np.sum(data, axis=1))
        
    def plot_cluster(self, data, cluster_id, type_id, title, max_ylim=None, ax=None):
        if title == "Elapsed":
            thresholds = self.quantization.e_thresholds
            split_num = self.quantization.e_quant_num
            labs = [f"~{thresholds[0]}[min]"] + [f"{thresholds[i-1]}~{thresholds[i]}" for i in range(1, len(thresholds))] + [f"{thresholds[-1]}~"]
        else:
            split_num = self.split_num
            labs = list(range(split_num))

        if max_ylim:
            ax.set_ylim(0, max_ylim)

        target_data = data[cluster_id]
        x = np.arange(self.quantization.dt_quant_num)
        x_labels = [str(int(i)*2) for i in range(self.quantization.dt_quant_num)]
        bottom = np.zeros(self.quantization.dt_quant_num)

        for split in range(split_num):
            y = target_data[type_id][split]
            ax.bar(x, y, bottom=bottom, color=self.color_list[split], label=labs[split])
            bottom += y

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=15)
        def y_formatter(y, pos):
            return f'{y:.3g}'

        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{ytick:.3g}' for ytick in ax.get_yticks()], fontsize=20)

        ax.set_title(f"Cluster: {cluster_id}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[::-1], labels=labels[::-1])
        

    def visualize(self, data, title, savefig=None):
        self.cluster_num = len(data)
        if 5*self.cluster_num*72 >= 2**16-1:
            fig, axes = plt.subplots(self.cluster_num, 2, figsize=(24, 900))
        else:
            fig, axes = plt.subplots(self.cluster_num, 2, figsize=(24, 5 * self.cluster_num))

        for cluster in range(self.cluster_num):
            max_ylim = self.get_max_value(data[cluster])

            ## Weekday
            ax_weekday = axes[cluster, 0]  # Select axis for Weekday
            self.plot_cluster(data, cluster, 0, title, max_ylim, ax=ax_weekday)

            ## Weekend
            ax_weekend = axes[cluster, 1]  # Select axis for Weekend
            self.plot_cluster(data, cluster, 1, title, max_ylim, ax=ax_weekend)

        if savefig:
            fig.savefig(savefig)

        fig.tight_layout()
        plt.show()

 
