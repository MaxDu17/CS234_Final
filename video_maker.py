import imageio

BASE_DIR = "experiments/800_step"
HARD_DIR = "experiments/1600_step"
hard_levels = [5, 8, 9, 11, 13, 15]

file_prefix = "full_algorithm_noanneal_level"

writer = imageio.get_writer("policy_evolution.mp4", fps=10)
rollout_writer = imageio.get_writer("rollouts.mp4", fps=20)

for i in range(18):
    print(i)
    for j in range(0, 800, 20):
        if i in hard_levels:
            image = f"{HARD_DIR}/{file_prefix}{i}_1_{2 * j}.png" #480, 640, 4)
        else:
            image = f"{BASE_DIR}/{file_prefix}{i}_1_{j}.png" #480, 640, 4)
        im = imageio.imread(image)
        writer.append_data(im)
    if i in hard_levels:
        try:
            im = imageio.get_reader(f"{HARD_DIR}/{file_prefix}{i}_1_{1600}.mp4")
        except:
            print("no video!")
            continue
    else:
        try:
            im = imageio.get_reader(f"{BASE_DIR}/{file_prefix}{i}_1_{800}.mp4")
        except:
            print("no video!")
            continue

    im_list = list(iter(im))
    for j in range(100):
        rollout_writer.append_data(im_list[j])


rollout_writer.close()
writer.close()

# im = imageio.get_reader("samples/sample.mp4")
# fps = im.get_meta_data()['fps']
# for frame in im:
#     print('reading and writing')
#     # print(frame.shape)
#     writer.append_data(frame[:, :, 1]) #grayscale rendering
# writer.close()
