LADCF_repo_path = '#LOCATION';

tracker_label = 'LADCF';
tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''LADCF'', ''VOT2018setting'', true)', {[LADCF_repo_path '/VOT_integration/benchmark_wrapper']});
tracker_interpreter = 'matlab';
