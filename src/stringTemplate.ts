export default function stringTemplate(str: string, obj: any) {
  

    const regex = /\${(\w+)}/gm;

    const res = regex.exec(str);

    res?.forEach(match => {
        
    });

    return str.replace(/\${(\w+)}/gm, )



}