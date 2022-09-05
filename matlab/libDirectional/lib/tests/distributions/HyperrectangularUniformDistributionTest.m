classdef HyperrectangularUniformDistributionTest < matlab.unittest.TestCase
    methods(Test)
        function testBasic(testCase)
            hud = HyperrectangularUniformDistribution([1,3;2,5]);
            [xMesh, yMesh] = meshgrid(linspace(1,3,50),linspace(2,5,50));
            testCase.verifyEqual(hud.pdf([xMesh(:)'; yMesh(:)']),1/6*ones(1,50^2));
        end
    end
end