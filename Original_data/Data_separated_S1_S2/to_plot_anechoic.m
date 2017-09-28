%to plot Anechoic%

load('A_N1_Tx_N2_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N1; Rx- N2; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N1; Rx- N2; Scenario:2')

%%
load('A_N1_Tx_N3_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N1; Rx- N3; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N1; Rx- N3; Scenario:2')

%%
load('A_N2_Tx_N1_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N2; Rx- N1; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N2; Rx- N1; Scenario:2')

%%
load('A_N2_Tx_N3_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N2; Rx- N3; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N2; Rx- N3; Scenario:2')

%%
load('A_N3_Tx_N1_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N3; Rx- N1; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N3; Rx- N1; Scenario:2')

%%
load('A_N3_Tx_N2_Rx_S1_S2.mat')

figure;    plot(P_dBm_out_n1_structure.vector_out_1,'b');   
title('Anechoic chamber; Tx- N3; Rx- N2; Scenario:1')    
figure;    plot(P_dBm_out_n1_structure.vector_out_2,'b');    
title('Anechoic chamber; Tx- N3; Rx- N2; Scenario:2')