import {useState} from 'react';
import {useRouter} from 'next/router';
import ExploreMain from '@/components/Explore/ExploreMain';

const Explore = () => {
  const router = useRouter();
  const [showFlow, setShowFlow] = useState(false);

  const handleClick = (sdgId: string) => {
    setShowFlow(true);
    router.push(`/explore/graph?sdgId=${sdgId}`);
  };

  return (
      <div>
        {!showFlow && <ExploreMain onButtonClick={handleClick}/>}
      </div>
  );
};

export default Explore;