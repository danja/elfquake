# ELFQuake

ELFQuake is an experimental project examining the viability of predicting earthquakes by means of Big Data and modern machine learning techniques. I started researching this several years ago. Recent advances in AI around pretrained models make this a good time to try it in practice. 
Historically earthquake prediction has the reputation of being a hard problem, typically only general trends are forecast. But it might now be feasible in such a way as to inform disaster prevention. The transformer architecture has shown itself remarkable at predicting sequences, most notably in large language models. It has also led to smart coding assistants that can significantly speed software prototyping. 

The approach taken is to expand the input data sources beyond the conventional seismic data to include natural (Very Low Frequency) radio signals and astronomical data, both of which have been considered to contain relevant information. However a prerequisite of large models is an abundance of data. This is not the case for VLF radio. The proposed workaround for this is to use a simulation to generate data of a similar shape to the missing VLF data (as well as associated synthetic seismic data).  

The work is ongoing, the code can be found at https://github.com/danja/elfquake 

Current status can be found in : https://github.com/danja/elfquake/blob/main/docs/report.md