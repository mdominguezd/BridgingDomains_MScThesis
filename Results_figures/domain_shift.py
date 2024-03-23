from Validate.domain_adaptation_performance import get_features_extracted, tSNE_source_n_target

def get_shift_measures(Love = False):
    """
        Function to retrieve the shift measures and tsne visualizations
    """
    if Love:

        DS_args = [16, None, False, None, None]

        cos_no = get_features_extracted(['urban'], ['rural'], DS_args, network_filename = 'BestModel.pt', Love = True, cos_int = True, euc_int = False)
        cos_da = get_features_extracted(['urban'], ['rural'], DS_args, network_filename = 'BestDANNModel_LOVE.pt', Love = True, cos_int = True, euc_int = False)

        print('Cosine similarity:\n No DA:' + str(cos_no) + '\nwith DA:' + str(cos_da))

        euc_no = get_features_extracted(['urban'], ['rural'], DS_args, network_filename = 'BestModel.pt', Love = True, cos_int = False, euc_int = True)
        euc_da = get_features_extracted(['urban'], ['rural'], DS_args, network_filename = 'BestDANNModel_LOVE.pt', Love = True, cos_int = False, euc_int = True)

        print('Euclidean distance:\n No DA:' + str(euc_no) + '\nwith DA:' + str(euc_da))

        
    else:
        DS_args = [16, None, 'Linear_1_99', True, False, None, None]
        
        cos_no = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', DS_args, network_filename = 'CIV_Fold1.pt', Love = False, cos_int = True, euc_int = False)
        cos_da = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', DS_args, network_filename = 'BestDANNModel_Cashew.pt', Love = False, cos_int = True, euc_int = False)
        
        print('Cosine similarity:\n No DA:' + str(cos_no) + '\nwith DA:' + str(cos_da))

        euc_no = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', DS_args, network_filename = 'CIV_Fold1.pt', Love = False, cos_int = False, euc_int = True)
        euc_da = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', DS_args, network_filename = 'BestDANNModel_Cashew.pt', Love = False, cos_int = False, euc_int = True)
        
        print('Euclidean distance:\n No DA:' + str(euc_no) + '\nwith DA:' + str(euc_da))

        