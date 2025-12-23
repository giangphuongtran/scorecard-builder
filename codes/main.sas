%let project_dir=\\10.200.43.8\Sandisk\credit-scoring;

libname raw "&project_dir\data\raw\inlib" compress=yes;
libname abt "&project_dir\data\work\abt" compress=yes;
libname models "&project_dir\models" compress=yes;
libname reports "&project_dir\reports" compress=yes;

%let input_data=raw.abt_app;
%let target_var=default12;