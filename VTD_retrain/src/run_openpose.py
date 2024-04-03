#!/usr/bin/python3

import os
import subprocess, signal
import time
import shutil

root_dir = os.getcwd()
# path where there are the images to process
dataset_dir = '/home/icub/shared/object-attention-dataset/ObjectAttention_augmentation'
# enter path where OpenPose was installed
os.chdir('/home/icub/code/openpose')
# path for openpose deployment command
openpose_path = './build/examples/openpose/openpose.bin'

save_openpose_images = False

#list_participants = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
#list_participants.sort()
list_participants = ['Giulia']

print("------- RUNNING OPENPOSE ON THE TRIALS ---------")
for participant in list_participants:
	print("Processing participant:", participant)
	participant_folder = os.path.join(dataset_dir, participant)
	list_sessions = [name for name in os.listdir(participant_folder) if os.path.isdir(os.path.join(participant_folder, name))]
	list_sessions = ['session_5']
	list_sessions.sort()
	print(list_sessions)
	for session in list_sessions:
		session_folder = os.path.join(participant_folder, session)
		if session == 'session_1':
			list_objects = [name for name in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, name))]
			list_objects.sort()
			print(list_objects)
			for obj in list_objects:
				obj_folder = os.path.join(session_folder, obj)
				list_rgb = [name for name in os.listdir(obj_folder) if os.path.isdir(os.path.join(obj_folder, name)) and 'rgb' in name and 'openpose' not in name]
				list_rgb.sort()
				print(list_rgb)
				for rgb in list_rgb:
					frames_folder = os.path.join(obj_folder, rgb)
					images_dir = frames_folder
					out_rend_dir = frames_folder + '_openpose_images'
					out_json_dir = frames_folder + '_openpose_json'

					if not os.path.exists(out_json_dir):
						os.makedirs(out_json_dir)

					if save_openpose_images:
						if not os.path.exists(out_rend_dir):
							os.makedirs(out_rend_dir)
								
					print("Processing %s" % frames_folder)
					# with image outputs from openpose
					if save_openpose_images:
						cmd = "%s --image_dir %s --display 0 --write_images %s --write_json %s --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
							% (openpose_path, images_dir, out_rend_dir, out_json_dir)
					else:
						cmd = "%s --image_dir %s --display 0 --render_pose 0 --write_json %s --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
							% (openpose_path, images_dir, out_json_dir)

					os.system(cmd)
					# pathh = shutil.which(openpose_path, mode=os.F_OK | os.X_OK, path=None)
					# print(pathh)
					#proc = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, close_fds=True)
					#print(proc)
					#time.sleep(60)
					#os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
					#subprocess.Popen.kill(proc)
					#else:
					#	print('OpenPose already run in the folder %s' % frames_folder)
		else:
			list_settings = [name for name in os.listdir(session_folder) if os.path.isdir(os.path.join(session_folder, name))]
			list_settings.sort()
			print(list_settings)
			for setting in list_settings:
				setting_folder = os.path.join(session_folder, setting)
				list_objects = [name for name in os.listdir(setting_folder) if os.path.isdir(os.path.join(setting_folder, name))]
				list_objects.sort()
				print(list_objects)
				for obj in list_objects:
					obj_folder = os.path.join(setting_folder, obj)
					list_rgb = [name for name in os.listdir(obj_folder) if os.path.isdir(os.path.join(obj_folder, name)) and 'rgb' in name and 'openpose' not in name]
					list_rgb.sort()
					print(list_rgb)
					for rgb in list_rgb:
						frames_folder = os.path.join(obj_folder, rgb)
						images_dir = frames_folder
						out_rend_dir = frames_folder + '_openpose_images'
						out_json_dir = frames_folder + '_openpose_json'

						if not os.path.exists(out_json_dir):
							os.makedirs(out_json_dir)

						if save_openpose_images:
							if not os.path.exists(out_rend_dir):
								os.makedirs(out_rend_dir)
								
						print("Processing %s" % frames_folder)
						# with image outputs from openpose
						if save_openpose_images:
							cmd = "%s --image_dir %s --display 0 --write_images %s --write_json %s --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
								% (openpose_path, images_dir, out_rend_dir, out_json_dir)
						else:
							cmd = "%s --image_dir %s --display 0 --render_pose 0 --write_json %s --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
								% (openpose_path, images_dir, out_json_dir)

						os.system(cmd)
						#proc = subprocess.run(cmd, shell=True, stdin=None, stdout=None, close_fds=True)
						#print(proc)
						#time.sleep(60)
						#os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
						#else:
						#	print('OpenPose already run in the folder %s' % frames_folder)

os.chdir(root_dir)

