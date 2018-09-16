function projection_matrix = init_projection_matrix(init_sample, compressed_dim, params)

% Initialize the projection matrix.

% Convert the compressed dimensions to a cell
compressed_dim_cell = permute(num2cell(compressed_dim), [2 3 1]);

% Reshape the sample
x = cellfun(@(x) reshape(x, [], size(x,3)), init_sample, 'uniformoutput', false);
x = cellfun(@(x) bsxfun(@minus, x, mean(x, 1)), x, 'uniformoutput', false);

[projection_matrix, ~, ~] = cellfun(@(x) svd(x' * x), x, 'uniformoutput', false);
projection_matrix = cellfun(@(P, dim) cast(P(:,1:dim), 'like', params.data_type), projection_matrix, compressed_dim_cell, 'uniformoutput', false);

end