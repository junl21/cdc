import torch
from torch.nn import functional as F


class TensorQueue(object):
    def __init__(self, k, n, dim):
        self.k = k
        self.n = n

        self.features_queue = torch.zeros((k, n, dim))
        self.queue_ptr = torch.zeros(k)

        self.current_size = torch.zeros(k)

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        for i in torch.unique(labels):
            batch_index = torch.where(labels == i)[0]
            update_size = batch_index.shape[0]
            queue_index = [int((x + self.queue_ptr[i]) % self.n) for x in range(update_size)]

            self.queue_ptr[i] = int((update_size + self.queue_ptr[i]) % self.n)
            self.features_queue[i, queue_index] = features[batch_index]

            self.current_size[i] = min(self.n, self.current_size[i] + update_size)

    def get_features(self):
        return self.features_queue.clone().detach()

    def load_queue(self, features, current_size):
        self.features_queue = features
        self.current_size = current_size

    def cuda(self, device=None):
        self.features_queue = self.features_queue.cuda(device)
        self.queue_ptr = self.queue_ptr.cuda(device)
        self.current_size = self.current_size.cuda(device)


class ClusterQueue(object):
    def __init__(self, k, n, dim, t_neg, t_pos, tmp):
        self.k = k
        self.n = n
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.tmp = tmp

        self.cluster_queue = TensorQueue(k=k, n=n, dim=dim)

    @torch.no_grad()
    def init_cluster(self, features, labels):
        features_all = concat_all_gather(features)
        labels_all = concat_all_gather(labels)

        self.cluster_queue.dequeue_and_enqueue(features_all, labels_all)

    @torch.no_grad()
    def calc_distance(self, x):
        x = F.normalize(x, dim=-1, p=2)
        cluster_features = self.cluster_queue.get_features().clone().detach()
        cluster_features = F.normalize(cluster_features, dim=-1, p=2)
        cluster_dis = torch.matmul(cluster_features, x.T).T
        cluster_dis = torch.sum(cluster_dis, dim=1) / self.cluster_queue.current_size
        return cluster_dis

    @torch.no_grad()
    def get_soft_label(self, features):
        cluster_dis = self.calc_distance(features)
        cluster_dis = cluster_dis / self.tmp
        return F.softmax(cluster_dis, dim=1)

    @torch.no_grad()
    def update(self, features, labels):
        # record current gpu for outputs
        batch_size = features.shape[0]
        rank_id = torch.distributed.get_rank()

        # gather from all gpus
        features_all = concat_all_gather(features)
        labels_all = concat_all_gather(labels)

        # record patch labels calculated from cluster
        cluster_targets = labels_all.clone()

        # calculate patch's probability belongs to each cluster
        soft_prob = self.get_soft_label(features_all)
        neg_prob = soft_prob[:, 0]

        update_ind = []
        # search negative patch and false positive patch
        neg_ind = torch.where(neg_prob > self.t_neg)[0]
        if neg_ind.shape[0] > 0:
            neg_update_ind = neg_ind[torch.where(labels_all[neg_ind] == 0)[0]]
            update_ind.append(neg_update_ind)

        # search positive patch
        pos_ind = torch.where(neg_prob < (1-self.t_pos))[0]
        if pos_ind.shape[0] > 0:
            # search positive patch originally belonged to positive cluster
            pos_update_ind = pos_ind[torch.where(labels_all[pos_ind] != 0)[0]]
            update_ind.append(pos_update_ind)
            # search hard negative patch originally belonged to negative cluster
            hard_neg_ind = pos_ind[torch.where(labels_all[pos_ind] == 0)[0]]
            update_ind.append(hard_neg_ind)

        # update cluster
        if len(update_ind) > 0:
            update_ind = torch.cat(update_ind)
            self.cluster_queue.dequeue_and_enqueue(features_all[update_ind], labels_all[update_ind])

        return cluster_targets[rank_id * batch_size: (rank_id + 1) * batch_size]

    def save_cluster(self, save_path):
        cluster_features = self.cluster_queue.get_features()
        cluster_checkpoint = {
            'features': cluster_features,
            'current_size': self.cluster_queue.current_size
        }
        torch.save(cluster_checkpoint, save_path)

    def load_cluster(self, data_path, device=None):
        cluster_checkpoint = torch.load(data_path, map_location='cpu')
        cluster_features = cluster_checkpoint['features'].cuda(device)
        current_size = cluster_checkpoint['current_size'].cuda(device)
        self.cluster_queue.load_queue(cluster_features, current_size)

    def cuda(self, device=None):
        self.cluster_queue.cuda(device)


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
