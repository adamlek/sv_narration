def sentence_level_eval(gold, pred):
    nc_sent = 0
    full = 0
    partial = 0
    none = 0
    rec_err = 0
    no_nc = 0
    for gsentence, psentence in zip(gold, pred):
        if 1 in gsentence:
            nc_sent += 1
            if gsentence == psentence:
                full += 1
            else:
                gset = set([i for (i,x) in enumerate(gsentence) if x == 1])
                pset = set([i for (i,x) in enumerate(psentence) if x == 1])
                if gset.intersection(pset):
                    partial += 1
                else:
                    none += 1
        else:
            no_nc += 1
            if 1 in psentence:
                rec_err += 1

    return full/nc_sent, partial/nc_sent, none/nc_sent, rec_err/no_nc
