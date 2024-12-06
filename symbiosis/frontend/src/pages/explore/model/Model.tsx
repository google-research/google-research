import {useRouter} from 'next/router';
import DrawFlowComponent from '@/components/Draw/DrawFlowComponent';

const Model = () => {
  const router = useRouter();
  const {sdgId, uuid, modelKey} = router.query;

  return (
      <DrawFlowComponent uuid={uuid} modelKey={modelKey}/>
  )
};


export default Model;