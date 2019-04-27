%% this is a demo of running DeepContext for testing

%% step 1: prepare data
cd code
script_data_prepare;
cd ..

%% step 2: scene template classification
cd scene_template_classification
demo_test_template_classification;
cd ..

%% step 3: transformation estimation
cd transformation
demo_transformation;
cd ..

%% step 4: 3d scene parsing
cd detection
demo_detection;
cd ..