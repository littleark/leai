import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Plus, Trash2 } from 'lucide-react';

export default function BookMetadataCollector() {
  const [metadata, setMetadata] = useState({
    reading_level: '',
    themes: [''],
    main_characters: [''],
    setting: '',
    genre: '',
    discussion_points: ['']
  });

  const addField = (field) => {
    setMetadata(prev => ({
      ...prev,
      [field]: [...prev[field], '']
    }));
  };

  const removeField = (field, index) => {
    setMetadata(prev => ({
      ...prev,
      [field]: prev[field].filter((_, i) => i !== index)
    }));
  };

  const updateField = (field, index, value) => {
    setMetadata(prev => ({
      ...prev,
      [field]: prev[field].map((item, i) => i === index ? value : item)
    }));
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4">
      <Card>
        <CardHeader className="bg-blue-50 border-b">
          <h3 className="text-lg font-semibold">📚 Book Metadata Collection</h3>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* Reading Level */}
          <div className="space-y-2">
            <label className="block font-medium">Reading Level</label>
            <select 
              className="w-full p-2 border rounded"
              value={metadata.reading_level}
              onChange={(e) => setMetadata(prev => ({...prev, reading_level: e.target.value}))}
            >
              <option value="">Select Reading Level</option>
              <option value="5-7">Ages 5-7</option>
              <option value="8-10">Ages 8-10</option>
              <option value="11-13">Ages 11-13</option>
            </select>
          </div>

          {/* Themes */}
          <div className="space-y-2">
            <label className="block font-medium">Themes</label>
            {metadata.themes.map((theme, index) => (
              <div key={index} className="flex gap-2">
                <input
                  type="text"
                  className="flex-1 p-2 border rounded"
                  value={theme}
                  onChange={(e) => updateField('themes', index, e.target.value)}
                  placeholder="Enter a theme"
                />
                {index > 0 && (
                  <button
                    onClick={() => removeField('themes', index)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => addField('themes')}
              className="flex items-center gap-2 text-blue-500 hover:bg-blue-50 p-2 rounded"
            >
              <Plus className="w-4 h-4" /> Add Theme
            </button>
          </div>

          {/* Characters */}
          <div className="space-y-2">
            <label className="block font-medium">Main Characters</label>
            {metadata.main_characters.map((char, index) => (
              <div key={index} className="flex gap-2">
                <input
                  type="text"
                  className="flex-1 p-2 border rounded"
                  value={char}
                  onChange={(e) => updateField('main_characters', index, e.target.value)}
                  placeholder="Enter a character name"
                />
                {index > 0 && (
                  <button
                    onClick={() => removeField('main_characters', index)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => addField('main_characters')}
              className="flex items-center gap-2 text-blue-500 hover:bg-blue-50 p-2 rounded"
            >
              <Plus className="w-4 h-4" /> Add Character
            </button>
          </div>

          {/* Discussion Points */}
          <div className="space-y-2">
            <label className="block font-medium">Discussion Points</label>
            {metadata.discussion_points.map((point, index) => (
              <div key={index} className="flex gap-2">
                <textarea
                  className="flex-1 p-2 border rounded"
                  value={point}
                  onChange={(e) => updateField('discussion_points', index, e.target.value)}
                  placeholder="Enter a discussion point or question"
                  rows="2"
                />
                {index > 0 && (
                  <button
                    onClick={() => removeField('discussion_points', index)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => addField('discussion_points')}
              className="flex items-center gap-2 text-blue-500 hover:bg-blue-50 p-2 rounded"
            >
              <Plus className="w-4 h-4" /> Add Discussion Point
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
