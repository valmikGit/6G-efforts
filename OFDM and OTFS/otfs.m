clc; clear; close all;

M = 64;             % No. of subcarriers
N = 1;              % No. of symbols
N_Frames = 10^4;    % Total No. of Frames

mod_order = 4;      % Modulation Order
bits_per_symbol = log2(mod_order);

symbols_per_frame = M * N;
bits_per_frame = symbols_per_frame * bits_per_symbol;

SNR_dB = -20:5:20;
SNR = 10.^(SNR_dB / 10);
noise_var_sqrt = sqrt(1 ./ SNR);

eng_sqrt = (mod_order == 2) + (mod_order ~= 2) * sqrt((mod_order - 1) / 6 * (2^2));
sigma_2 = abs(eng_sqrt * noise_var_sqrt).^2;

rng(1)
err_ber = zeros(length(SNR_dB), 1);

% Information Bits Generation
bi_data = randi([0 1], bits_per_frame * N_Frames, 1);
data_mtx = reshape(bi_data, bits_per_symbol, [])';
de_data = bi2de(data_mtx, 'left-msb');
de_data = reshape(de_data, symbols_per_frame, N_Frames);

x = qammod(de_data, mod_order, 'gray');
X = reshape(x, M, N, N_Frames);

% OTFS Modulation
F_M = dftmtx(M);
F_N = dftmtx(N);
Gtx = eye(M);
Grx = Gtx;

S = pagemtimes(Gtx, pagemtimes(X, permute(F_N, [2, 1, 3])));
s = reshape(S, [], N_Frames);

% Channel Generation
taps = 4;
Delay_taps = 0:taps-1;
Doppler_taps = 0:taps-1;
pow_prof = (1 / taps) .* ones(1, taps);
h = sqrt(pow_prof).' .* (sqrt(1 / 2) * (randn(taps, N_Frames) + 1i * randn(taps, N_Frames)));

% Channel Matrix Computation
I = eye(M * N);
z = exp((1i * 2 * pi) / (M * N));
Del = diag(z .^ (0:((M * N) - 1)));
H = zeros(M * N, M * N, N_Frames);

for f = 1:N_Frames
    H_temp = zeros(M * N, M * N);
    for t = 1:taps
        H_temp = H_temp + h(t, f) * circshift(I, Delay_taps(t)) * (Del ^ Doppler_taps(t));
    end
    H(:,:,f) = H_temp;
end

% Noise Generation
for snr_idx = 1:length(SNR_dB)
    w = sqrt(sigma_2(snr_idx) / 2) * (randn(M * N, N_Frames) + 1i * randn(M * N, N_Frames));

    % Receiver Processing
    w_tilda = pagemtimes(kron(F_N, Grx), w);
    Heff = pagemtimes(kron(F_N, Grx), pagemtimes(H, kron(F_N', Gtx)));
    y = pagemtimes(Heff, s) + w_tilda;

    % Use PAGEPINV instead of PINV to handle 3D matrices
    Heff_inv = pagepinv(Heff);
    x_cap = pagemtimes(Heff_inv, y);

    de_data_cap = qamdemod(x_cap, mod_order, 'gray');
    de_data_cap = round(real(de_data_cap));
    data_mtx_cap = de2bi(de_data_cap, bits_per_symbol, 'left-msb');
    
    % Ensure correct reshaping
    bi_data_est = reshape(data_mtx_cap', [], 1);
    bi_data_est = bi_data_est(1:length(bi_data)); % Match original bit length
    
    error = sum(xor(bi_data_est, bi_data));
    err_ber(snr_idx) = error / length(bi_data);
end

% Plotting
semilogy(SNR_dB, err_ber, '-*', 'LineWidth', 2);
title('OTFS');
ylabel('BER'); xlabel('SNR in dB'); grid on;