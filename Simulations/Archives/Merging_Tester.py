# import h5py

# with h5py.File("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5", 'r') as h5fr:
#     # for link in h5fr:
#     #     print(link)
#     #     f = h5fr[link]

#     #     random_mags_dset = f["random_mags dataset"]
#     #     random_angs_dset = f["random_angs dataset"]
#     #     random_I_dset = f["random_I dataset"]
#     #     random_t_dset = f["random_t dataset"]
#     #     ronch_dset = f["ronch dataset"]

#     #     for dset in [random_mags_dset, random_angs_dset, random_I_dset, random_t_dset, ronch_dset]:
#     #         print("\n", dset)
#     #         print(dset[49, :])

#     random_mags_dset_list = []
#     random_angs_dset_list = []
#     random_I_dset_list = []
#     random_t_dset_list = []
#     ronch_dset_list = []

#     for link in h5fr:
#         f = h5fr[link]

#         random_mags_dset_list.append(f["random_mags dataset"])
#         random_angs_dset_list.append(f["random_angs dataset"])
#         random_I_dset_list.append(f["random_I dataset"])
#         random_t_dset_list.append(f["random_t dataset"])
#         ronch_dset_list.append(f["ronch dataset"])

#     for i in range(len(random_mags_dset_list)):
#         for dset in [random_mags_dset_list[i], random_angs_dset_list[i], random_I_dset_list[i], random_t_dset_list[i], ronch_dset_list[i]]:
#             print("\n", dset)
#             print(dset[49, :])

#     print(len(random_angs_dset_list))
#     print(len(random_angs_dset_list[0]))


# h5fr = h5py.File("/media/rob/hdd1/james-gj/Ronchigrams/Simulations/08_12_2021/links.h5", 'r')
# for link in h5fr:
#     print(link)
#     f = h5fr[link]

#     random_mags_dset = f["random_mags dataset"]
#     random_angs_dset = f["random_angs dataset"]
#     random_I_dset = f["random_I dataset"]
#     random_t_dset = f["random_t dataset"]
#     ronch_dset = f["ronch dataset"]

#     for dset in [random_mags_dset, random_angs_dset, random_I_dset, random_t_dset, ronch_dset]:
#         print("\n", dset)
#         print(dset[49, :])
# h5fr.close()

arr = 2
if arr == 2:
    print("yes")