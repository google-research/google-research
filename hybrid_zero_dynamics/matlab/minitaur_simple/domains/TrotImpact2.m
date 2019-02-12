% Copyright 2019 Google LLC
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function guard = TrotImpact2(domain, load_path)
% Trot Impact Two Impact (guard)
% Set impact
guard = RigidImpact('TrotImpact2', domain, 'frontrightFootHeight');


% Set the impact constraint
% we will compute the impact map every time you add an impact
% constraints, so it would be helpful to load expressions directly
guard.addImpactConstraint(struct2array(domain.HolonomicConstraints), load_path);
end