function CubObj = Rectangle(ln,bd,ht,R,center,color)

    % transparency value
    alph = 0.5;

    %% Create Vertices
    x = 0.5*ln*[-1 1 1 -1 -1 1 1 -1]';
    y = 0.5*bd*[1 1 1 1 -1 -1 -1 -1]';
    z = 0.5*ht*[-1 -1 1 1 1 1 -1 -1]';

    %% Create Faces
    facs = [1 2 3 4
            5 6 7 8
            4 3 6 5
            3 2 7 6
            2 1 8 7
            1 4 5 8];
    %% Rotate and Translate Vertices
        verts = zeros(3,8);
        for i = 1:8
            verts(1:3,i) = R*[x(i);y(i);z(i)]+R*[center(1);center(2);center(3)];
        end    

    CubObj = patch('Faces',facs,'Vertices',verts','FaceColor', color,'FaceAlpha',alph);

end