def get_rank(rank, ngpus_per_node):
    return rank % ngpus_per_node


def is_rank0_process(multiprocessing_distributed):
    return not multiprocessing_distributed or (
        multiprocessing_distributed and get_rank() == 0
    )
