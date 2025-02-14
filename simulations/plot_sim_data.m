% Function to plot the simulated curvature, curvature simulations, and estimated
% discriminability matrix.
%
%   x                        - [n_dim x n_frames]     perceptual locations
%   c_true                   - [n_frames - 2]         ground truth curvatures
%   c_est                    - [n_frames - 2]         estimated curvatures 
%   discriminability_matrix  - [n_frames  x n_frames] discriminability matrix, each number corresponds to the percentage 
%                                                     of correct trials for the respective condition 

function plot_sim_data(x, c_true, c_est, discriminability_mat, n_reps)

    n_frames = size(discriminability_mat, 1);

    subplot(1,3,1)
    plot(x(1, :), x(2, :), 'ko-', 'markersize', 12, 'markerfacecolor', [1 0 0], 'linewidth', 1)
    hold on, box off, axis equal, axis square

    subplot(1,3,2)
    plot(c_true, c_est, 'r+', 'linewidth', 1)
    hold on, box off, axis square
    plot([0 pi], [0 pi], 'k--')
    plot(mean(c_true), mean(c_est), 'ko', 'MarkerSize', 12, 'markerfacecolor', [0 .5 1])

    subplot(1,3,3)
    imagesc(discriminability_mat);
    hold on, axis square, box off
    colormap('gray'); 
    cBar = colorbar; 
    title(cBar,'Proportion correct'); 
    cBar.Ticks = [0.35,1];
    xticks([1 n_frames]); yticks([1 n_frames]);
    xlim([1 n_frames]); ylim([1 n_frames]);
    xlabel('Frame number'); ylabel('Frame number');
    titleText = sprintf('Simulated discriminability, %d trials', n_reps);
    title(titleText);
    set(gca, 'FontName', 'Arial');
    set(gca, 'FontSize', 12);
end