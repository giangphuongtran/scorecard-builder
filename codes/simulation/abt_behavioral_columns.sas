%macro make_abt(period);

data period;
periodp=input("&period", yymm6.);
do i = 0 to &max_length - 1;
period = put(intnx('month', periodp, -i, 'end'), yymm6.);
output;
end;
keep period;
run;

proc sql noprint;
select period
into :periods separated by ' '
from periods order by 1;
quit;
%let n_periods = &max_length;
%put &n_periods;
%put &periods;

%let first_period = %scan(&periods, 1, %str( ));
%put &first_period;

data _null_;
index = intck('month', input("&first_period", yymm6.), input("&period", yymm6.)) + 1;
call symput('input', put(index, best12. - L));
run;
%put &index;

%let var1 = CMaxI_Days;
%let var2 = = CMaxI_Due;
%let var3 = CMaxC_Days;
%let var4 = CMaxC_Due;
%let var5 = CMaxA_Days;
%let var6 = CMaxA_due;

%let des1 = Maximum Customer days for Ins product;
%let des2 = Maximum Customer due for Ins product;
%let des3 = Maximum Customer days for Css product;
%let des4 = Maximum Customer due for Css product;
%let des5 = Maximum Customer days for all products;
%let des6 = Maximum Customer due for all products;

%let n_var_agr = 6;

%let sagr1 = Mean;
%let sagr2 = Max;
%let sagr3 = Min;
%let n_sagr = 3;

%let lengths = 3 6 9 12;
%let n_lengths = 4;

data &data_wyj;
array tx(&max_length);
array ty(&max_length);

set &data_wej;

%do len = 1 to &n_lengths;
%let length = %scan(&lengths, &lengths, %str( ));
%let first_index = %eval(&index - &length + 1);
%if &first_index < 1 %then %let first_index = 1;
	%do v = 1 %to &n_var_agr;
		%do a = 1 %to &n_sagr;
			agr&length._&&sagr&aabort._&&var&v=&&sagr&a(
			%do i = &first_index %to &index;
				%let p = %scan(&periods, &i, %str( ));
				&&var&v.._&p,
				%end;
			.);
			nmiss = nmiss(
			%do i = &first_index % to &index;
				%let p = %scan(&periods, &i, %str( ));
				&&var&v.._&p,
				%end;
			.);
			ags&length._&&sagr&a.._&&var&v=agr&length._&&sagr&a.._&&var&v;
			if nmiss>1 then agr&length._&&sagr&a.._&&var&v=.m;
			label ags&length._&&sagr&a.._&&var&v=
				"&&sagr&a.. calculated on last &length. months on &&des&v";
			label agr&length._&&sagr&a.._&&var&v=
				"&&sagr&a.. calculated on last &length. months on unmissing &&des&v ";
		%end;
	%end;

	act&length._n_arrears = sum(
		%do i = &first_index %to &index;
			%let p = %scan(&periods, &i, %str( ));
			(CMaxA_Due_&p >= 1),
			%end;
		.);
	label act&length._n_arrears = "Customer number in arrears on all loans";
	
	act&length._n_arrears = sum(
		%do i = &first_index %to &index;
			%let p = %scan(&periods, &i, %str( ));
			(CMaxA_Due_&p > 15),
			%end;
		.);
	label act&length._n_arrears = "Customer number of days greater than 15 on all loans";

	act&length._n_good_days = sum(
		%do i = &first_index %to &index;
			%let p = %scan(&periods, &i, %str( ));
			(CMaxA_Due_&p < 1),
			%end;
		.);
	label act&length._n_arrears_days = "Customer number of days lower than 15 on all loans";

%end;

if _error_ = 1 then _error_ = 0;
keep &id_account agr: act: ags:;
run;

%mend;