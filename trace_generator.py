import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate some randomly distributed data
sigma_fs=10e3
mu_fs=10e3
size_dataset=1000


# Generate flow size distribution
flow_size_training = np.round(np.abs(mu_fs+sigma_fs*np.random.randn(size_dataset)))
flow_size_test = np.round(np.abs(mu_fs+sigma_fs*np.random.randn(size_dataset)))
flow_size_validation = np.round(np.abs(mu_fs+sigma_fs*np.random.randn(size_dataset)))


#First call size feature
sigma_first_call=30
first_call_training=np.round(np.abs(100*flow_size_training/mu_fs+sigma_first_call*np.random.randn(size_dataset)))
first_call_test=np.round(np.abs(100*flow_size_test/mu_fs+sigma_first_call*np.random.randn(size_dataset)))
first_call_validation=np.round(np.abs(100*flow_size_validation/mu_fs+sigma_first_call*np.random.randn(size_dataset)))

#CPU IO feature
sigma_CPU_call=50
CPU_training=np.round(np.abs(200*flow_size_training/mu_fs+sigma_CPU_call*np.random.randn(size_dataset)))
CPU_test=np.round(np.abs(200*flow_size_test/mu_fs+sigma_CPU_call*np.random.randn(size_dataset)))
CPU_validation=np.round(np.abs(200*flow_size_validation/mu_fs+sigma_CPU_call*np.random.randn(size_dataset)))

#Disk IO feature
sigma_disk=30
Disk_training=np.round(np.abs(50*flow_size_training/mu_fs+sigma_disk*np.random.randn(size_dataset)))
Disk_test=np.round(np.abs(50*flow_size_test/mu_fs+sigma_disk*np.random.randn(size_dataset)))
Disk_validation=np.round(np.abs(50*flow_size_validation/mu_fs+sigma_disk*np.random.randn(size_dataset)))



#define dataframe's features 
df_training = pd.DataFrame({'flow_size': flow_size_training,'first_call_size':first_call_training, 'CPU_IOS':CPU_training,'Disk_IO':Disk_training})

df_test = pd.DataFrame({'flow_size': flow_size_test, 'first_call_size':first_call_test, 'CPU_IOS':CPU_test,'Disk_IO':Disk_test})

df_validation = pd.DataFrame({'flow_size': flow_size_validation, 'first_call_size':first_call_validation, 'CPU_IOS':CPU_validation,'Disk_IO':Disk_validation})


#csv paths
csv_training = './data/training/training_data.csv'
csv_test = './data/test/test_data.csv'
csv_validation = './data/validation/validation_data.csv'

#Write to csvs
df_training.to_csv(csv_training, index=False)
df_test.to_csv(csv_test, index=False)
df_validation.to_csv(csv_validation, index=False)

'''
#print('len data',len(data))
df_training = pd.DataFrame({'flow_size': flow_size_training})
csv_training = './data/training/training_data.csv'
df_training.to_csv(csv_training, index=False)

df_test = pd.DataFrame({'flow_size': flow_size_test})
csv_test = './data/test/test_data.csv'
df_test.to_csv(csv_test, index=False)

df_validation = pd.DataFrame({'flow_size': flow_size_validation})
csv_validation = './data/validation/validation_data.csv'
df_validation.to_csv(csv_validation, index=False)
'''



#print('\ndata_training\n',data_training)
# Sort the data
data_training_sorted = np.sort(flow_size_training)
data_test_sorted = np.sort(flow_size_test)
data_validation_sorted = np.sort(flow_size_validation)


# Calculate the proportional values of samples
p_training = np.linspace(0, 1, len(flow_size_training))
p_test = np.linspace(0, 1, len(flow_size_test))
p_validation = np.linspace(0, 1, len(flow_size_validation))

# Plot the sorted data and the CDF
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
ax1.plot(data_training_sorted, p_training)
ax1.set_xlabel("$data training$")
ax1.set_ylabel('$p$')
ax1.set_title("Cumulative Distribution Function (CDF) data_training")
ax1.set_xscale("log")
ax1.grid()

ax2.plot(data_test_sorted, p_test)
ax2.set_xlabel('$data test$')
ax2.set_ylabel('$p$')
ax2.set_title("Cumulative Distribution Function (CDF) data_test")
ax2.set_xscale("log")
ax2.grid()

ax3.plot(data_validation_sorted, p_validation)
ax3.set_xlabel('$data validation$')
ax3.set_ylabel('$p$')
ax3.set_title("Cumulative Distribution Function (CDF) data_validation")
ax3.set_xscale("log")
ax3.grid()

plt.tight_layout()
plt.show()
