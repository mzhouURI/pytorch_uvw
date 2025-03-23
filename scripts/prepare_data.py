import rosbag
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
import pandas as pd

bag_file = '../bags/2025-03-13-14-43-13.bag'

thruster_topics = ['/alpha_rise/control/thruster/surge',
                   '/alpha_rise/control/thruster/sway_bow',
                   '/alpha_rise/control/thruster/heave_bow',
                   '/alpha_rise/control/thruster/heave_stern']

# Initialize a dictionary to store data from different topics
thruster_data = {topic: [] for topic in thruster_topics}


##read odom
twist_topic = '/alpha_rise/odometry/filtered/local'
dvl_topic = '/alpha_rise/dvl/twist'
odom_twist = []



# Open the bag file and read the messages
with rosbag.Bag(bag_file, 'r') as bag:
    start_time = bag.get_start_time()  # returns float seconds since epoch

    # Iterate through the messages in the specified topics
    for topic, msg, t in bag.read_messages(topics=thruster_topics):
        # Append the timestamp and message data to the corresponding topic's list
        thruster_data[topic].append([t.to_sec() - start_time, msg.data])

    for topic, msg, t in bag.read_messages(topics=[dvl_topic]):
        timestamp = msg.header.stamp.to_sec() - start_time
        twist = msg.twist.twist  ##for odom
        odom_twist.append([timestamp, twist.linear.x, twist.linear.y, twist.linear.z])
        # else:
            # print(f"Skipping message of type {type(msg)}")  # Print the type if it's not Odometry

#manually remove the first entry of surge, sway and HB 
# thruster_data[thruster_topics[0]] = thruster_data[thruster_topics[0]][1:]
# thruster_data[thruster_topics[1]] = thruster_data[thruster_topics[1]][1:]
# thruster_data[thruster_topics[2]] = thruster_data[thruster_topics[2]][1:]

    
surge_data = np.array(thruster_data[thruster_topics[0]])
sway_data = np.array(thruster_data[thruster_topics[1]])
hb_data = np.array(thruster_data[thruster_topics[2]])
hs_data = np.array(thruster_data[thruster_topics[3]])


odom_twist_array = np.array(odom_twist)

# print(odom_twist_array.shape)

# plt.figure(figsize=(10, 6))

# # Plot each component
# plt.plot(odom_twist_array[:, 0], odom_twist_array[:, 1], label='Linear X', color='r')  # linear x
# plt.plot(odom_twist_array[:, 0], odom_twist_array[:, 2], label='Linear Y', color='g')  # linear y
# plt.plot(odom_twist_array[:, 0], odom_twist_array[:, 3], label='Linear Z', color='b')  # linear z
# plt.show()



#interpolation

odom_timestamps = odom_twist_array[:, 0]
surge_timestamps = surge_data[:, 0]
sway_timestamps = sway_data[:, 0]
hb_timestamps = hb_data[:, 0]
hs_timestamps = hs_data[:, 0]

# Values
surge_values = surge_data[:, 1]
sway_values = sway_data[:, 1]
hb_values = hb_data[:, 1]
hs_values = hs_data[:, 1]

# Interpolation
interpolated_surge = np.interp(odom_timestamps, surge_timestamps, surge_values)
interpolated_sway = np.interp(odom_timestamps, sway_timestamps, sway_values)
interpolated_hb = np.interp(odom_timestamps, hb_timestamps, hb_values)
interpolated_hs = np.interp(odom_timestamps, hs_timestamps, hs_values)

# Combine into a single array
interpolated_thruster_data = np.vstack([interpolated_surge, interpolated_sway, interpolated_hb, interpolated_hs]).T

# Add odom timestamps as the first column if needed
final_data = np.column_stack([odom_twist_array, interpolated_thruster_data])

print(final_data.shape)

np.savetxt('test_data.csv', final_data, delimiter=',', header='timestamp,u,v,w,surge,sway,hb,hs', comments='', fmt='%f')


# Plotting
plt.figure(figsize=(10, 6))

# # Plot interpolated surge data
# plt.plot(final_data[:,1], final_data[:,1],  color='blue')
# plt.plot(final_data[:, 4], final_data[:, 1], color='red', marker='o', linestyle='')
plt.plot(final_data[:, 0], final_data[:, 4], color='red', marker='o', linestyle='')
plt.plot(surge_data[:, 0], surge_data[:, 1], color='blue', marker='o', linestyle='')

plt.show()