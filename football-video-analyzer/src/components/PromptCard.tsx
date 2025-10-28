'use client';

interface PromptCardProps {
  title: string;
  value: string;
  onChange: (value: string) => void;
  maxLength: number;
  placeholder?: string;
  rows?: number;
}

export default function PromptCard({ 
  title, 
  value, 
  onChange, 
  maxLength, 
  placeholder = "",
  rows = 4 
}: PromptCardProps) {
  const remainingChars = maxLength - value.length;
  const isNearLimit = remainingChars < 50;

  return (
    <div className="nvidia-card">
      <h3 className="nvidia-card-header">{title}</h3>
      
      <textarea
        className={`nvidia-input ${isNearLimit ? 'border-red-500' : ''}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        maxLength={maxLength}
      />
      
      <div className={`char-counter ${isNearLimit ? 'text-red-400' : ''}`}>
        {value.length}/{maxLength}
      </div>
    </div>
  );
}
